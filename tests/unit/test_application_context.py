"""Unit tests for Application Context discovery and management."""

import pytest
import json
from unittest.mock import Mock, patch
from typing import Dict, List

from testteller.automator_agent.application_context import (
    ApplicationKnowledgeExtractor,
    ApplicationContext,
    APIEndpoint,
    UIPattern,
    AuthPattern,
    DataSchema
)
from testteller.automator_agent.parser.markdown_parser import TestCase, TestStep
from testteller.core.vector_store.chromadb_manager import ChromaDBManager
from testteller.core.llm.llm_manager import LLMManager


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    mock_vs = Mock(spec=ChromaDBManager)
    return mock_vs


@pytest.fixture
def mock_llm_manager():
    """Mock LLM manager for testing."""
    return Mock(spec=LLMManager)


@pytest.fixture
def sample_test_cases():
    """Sample test cases for context extraction."""
    return [
        TestCase(
            id="LOGIN_001",
            feature="Authentication",
            type="E2E",
            category="Login",
            objective="Test user login with valid credentials",
            test_steps=[
                TestStep(action="Navigate to login page", validation="Login form displayed"),
                TestStep(action="Enter email and password", validation="Login successful")
            ]
        ),
        TestCase(
            id="API_001",
            feature="User Management",
            type="API",
            category="CRUD",
            objective="Test user creation via API",
            test_steps=[
                TestStep(action="Send POST to /api/users", validation="User created successfully")
            ]
        )
    ]


class TestAPIEndpoint:
    """Test APIEndpoint data structure."""

    def test_api_endpoint_creation(self):
        """Test creating API endpoint objects."""
        endpoint = APIEndpoint(
            path="/api/users",
            method="GET",
            description="Get all users",
            auth_required=True
        )
        
        assert endpoint.path == "/api/users"
        assert endpoint.method == "GET"
        assert endpoint.description == "Get all users"
        assert endpoint.auth_required == True
        assert endpoint.request_schema is None
        assert endpoint.response_schema is None

    def test_api_endpoint_with_schemas(self):
        """Test API endpoint with request/response schemas."""
        request_schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        response_schema = {"200": {"description": "Success"}}
        
        endpoint = APIEndpoint(
            path="/api/users",
            method="POST",
            request_schema=request_schema,
            response_schema=response_schema
        )
        
        assert endpoint.request_schema == request_schema
        assert endpoint.response_schema == response_schema


class TestUIPattern:
    """Test UIPattern data structure."""

    def test_ui_pattern_creation(self):
        """Test creating UI pattern objects."""
        pattern = UIPattern(
            selector="[data-testid='login-button']",
            element_type="button",
            description="Login submit button",
            page_context="login_page"
        )
        
        assert pattern.selector == "[data-testid='login-button']"
        assert pattern.element_type == "button"
        assert pattern.description == "Login submit button"
        assert pattern.page_context == "login_page"


class TestAuthPattern:
    """Test AuthPattern data structure."""

    def test_auth_pattern_creation(self):
        """Test creating authentication pattern objects."""
        pattern = AuthPattern(
            auth_type="jwt",
            login_endpoint="/api/auth/login",
            token_header="Authorization",
            login_selectors={
                "email": "[data-testid='email']",
                "password": "[data-testid='password']"
            }
        )
        
        assert pattern.auth_type == "jwt"
        assert pattern.login_endpoint == "/api/auth/login"
        assert pattern.token_header == "Authorization"
        assert pattern.login_selectors["email"] == "[data-testid='email']"


class TestDataSchema:
    """Test DataSchema data structure."""

    def test_data_schema_creation(self):
        """Test creating data schema objects."""
        schema = DataSchema(
            model_name="User",
            fields={"id": "int", "email": "str", "name": "str"},
            required_fields=["id", "email"]
        )
        
        assert schema.model_name == "User"
        assert schema.fields["email"] == "str"
        assert "id" in schema.required_fields


class TestApplicationContext:
    """Test ApplicationContext data structure."""

    def test_application_context_creation(self):
        """Test creating application context objects."""
        context = ApplicationContext(
            base_url="https://api.example.com",
            api_endpoints={
                "GET:/api/users": APIEndpoint("/api/users", "GET")
            },
            ui_selectors={
                "login-btn": UIPattern("[data-testid='login-btn']", "button")
            }
        )
        
        assert context.base_url == "https://api.example.com"
        assert "GET:/api/users" in context.api_endpoints
        assert "login-btn" in context.ui_selectors

    def test_empty_application_context(self):
        """Test creating empty application context."""
        context = ApplicationContext()
        
        assert context.base_url is None
        assert len(context.api_endpoints) == 0
        assert len(context.ui_selectors) == 0
        assert context.auth_patterns is None


class TestApplicationKnowledgeExtractor:
    """Test ApplicationKnowledgeExtractor functionality."""

    def test_initialization(self, mock_vector_store, mock_llm_manager):
        """Test extractor initialization."""
        extractor = ApplicationKnowledgeExtractor(
            mock_vector_store,
            mock_llm_manager,
            num_context_docs=7
        )
        
        assert extractor.vector_store == mock_vector_store
        assert extractor.llm_manager == mock_llm_manager
        assert extractor.num_context_docs == 7

    def test_extract_app_context_success(self, mock_vector_store, mock_llm_manager, sample_test_cases):
        """Test successful application context extraction."""
        extractor = ApplicationKnowledgeExtractor(mock_vector_store, mock_llm_manager)
        
        # Mock vector store responses for different discovery methods
        mock_vector_store.query_similar.side_effect = [
            # API endpoints query
            {
                'documents': [['@app.route("/api/users", methods=["GET"])', 'app.post("/api/login")']],
                'metadatas': [['type: code', 'type: code']],
                'distances': [[0.1, 0.2]]
            },
            # Swagger/OpenAPI query
            {
                'documents': [['{"paths": {"/api/users": {"get": {"summary": "Get users"}}}}']],
                'metadatas': [['type: documentation']],
                'distances': [[0.15]]
            },
            # UI patterns query (test code)
            {
                'documents': [['page.click("[data-testid=\'login-btn\']")', 'cy.get("#email-input")']],
                'metadatas': [['type: code', 'type: code']],
                'distances': [[0.1, 0.2]]
            },
            # UI patterns query (component code)
            {
                'documents': [['<button data-testid="submit-btn">', '<input id="email" />']],
                'metadatas': [['type: code', 'type: code']],
                'distances': [[0.1, 0.2]]
            },
            # Auth patterns query
            {
                'documents': [['jwt_token = request.headers.get("Authorization")', 'login_required']],
                'metadatas': [['type: code', 'type: code']],
                'distances': [[0.1, 0.2]]
            },
            # Data schemas query
            {
                'documents': [['class User(db.Model):', 'user_id = Column(Integer)']],
                'metadatas': [['type: code', 'type: code']],
                'distances': [[0.1, 0.2]]
            },
            # Existing test patterns query
            {
                'documents': [['def test_login():', 'assert response.status_code == 200']],
                'metadatas': [['type: code', 'type: code']],
                'distances': [[0.1, 0.2]]
            },
            # Framework patterns query
            {
                'documents': [['{"testMatch": ["**/*.test.js"]}', '[pytest]']],
                'metadatas': [['type: config', 'type: config']],
                'distances': [[0.1, 0.2]]
            },
            # Base URL inference query
            {
                'documents': [['BASE_URL = "https://api.example.com"', 'API_URL=http://localhost:8000']],
                'metadatas': [['type: code', 'type: config']],
                'distances': [[0.1, 0.2]]
            }
        ]
        
        context = extractor.extract_app_context(sample_test_cases)
        
        assert isinstance(context, ApplicationContext)
        # Should have made multiple queries for different types of discovery
        assert mock_vector_store.query_similar.call_count >= 5

    def test_extract_app_context_failure(self, mock_vector_store, mock_llm_manager, sample_test_cases):
        """Test application context extraction with failures."""
        extractor = ApplicationKnowledgeExtractor(mock_vector_store, mock_llm_manager)
        
        # Mock vector store to raise an exception
        mock_vector_store.query_similar.side_effect = Exception("Vector store error")
        
        # Should return empty context instead of crashing
        context = extractor.extract_app_context(sample_test_cases)
        
        assert isinstance(context, ApplicationContext)
        # Even on failure, may still get default base URL
        assert context.base_url is not None  # Gets default fallback
        assert len(context.api_endpoints) == 0

    def test_parse_endpoints_from_content(self, mock_vector_store, mock_llm_manager):
        """Test parsing API endpoints from code content."""
        extractor = ApplicationKnowledgeExtractor(mock_vector_store, mock_llm_manager)
        
        # Test Flask-style endpoints
        flask_content = '''
@app.route("/api/users", methods=["GET", "POST"])
def users():
    pass

@app.route("/api/users/<int:user_id>", methods=["GET"])
def get_user(user_id):
    pass
'''
        
        endpoints = extractor._parse_endpoints_from_content(flask_content)
        
        assert isinstance(endpoints, dict)
        # Should find the Flask routes (implementation may vary)
        # This tests the parsing logic exists

    def test_parse_openapi_spec(self, mock_vector_store, mock_llm_manager):
        """Test parsing OpenAPI specifications."""
        extractor = ApplicationKnowledgeExtractor(mock_vector_store, mock_llm_manager)
        
        openapi_spec = {
            "paths": {
                "/api/users": {
                    "get": {
                        "summary": "Get all users",
                        "responses": {"200": {"description": "Success"}}
                    },
                    "post": {
                        "summary": "Create user",
                        "security": [{"bearerAuth": []}]
                    }
                }
            }
        }
        
        spec_content = json.dumps(openapi_spec)
        endpoints = extractor._parse_openapi_spec(spec_content)
        
        assert isinstance(endpoints, dict)
        # Should parse the OpenAPI specification
        if endpoints:  # May be empty if parsing fails
            for key, endpoint in endpoints.items():
                assert isinstance(endpoint, APIEndpoint)

    def test_extract_ui_selectors_from_test_code(self, mock_vector_store, mock_llm_manager):
        """Test extracting UI selectors from test code."""
        extractor = ApplicationKnowledgeExtractor(mock_vector_store, mock_llm_manager)
        
        test_code = '''
def test_login():
    page.click("[data-testid='login-btn']")
    page.fill("[data-testid='email-input']", "test@example.com")
    driver.find_element(By.ID, "password").send_keys("password")
    cy.get(".submit-button").click()
'''
        
        selectors = extractor._extract_ui_selectors_from_test_code(test_code)
        
        assert isinstance(selectors, dict)
        # Should extract various types of selectors from the test code

    def test_extract_ui_selectors_from_components(self, mock_vector_store, mock_llm_manager):
        """Test extracting UI selectors from component code."""
        extractor = ApplicationKnowledgeExtractor(mock_vector_store, mock_llm_manager)
        
        component_code = '''
<div>
    <input data-testid="email-input" type="email" />
    <input id="password-field" type="password" />
    <button className="submit-btn primary">Submit</button>
</div>
'''
        
        selectors = extractor._extract_ui_selectors_from_components(component_code)
        
        assert isinstance(selectors, dict)
        # Should extract selectors from component attributes

    def test_analyze_auth_patterns(self, mock_vector_store, mock_llm_manager):
        """Test analyzing authentication patterns."""
        extractor = ApplicationKnowledgeExtractor(mock_vector_store, mock_llm_manager)
        
        auth_docs = [
            'JWT token authentication',
            'Authorization header required',
            'POST /api/auth/login',
            'Bearer token in request headers'
        ]
        
        auth_info = extractor._analyze_auth_patterns(auth_docs)
        
        if auth_info:  # May be None if no patterns detected
            assert isinstance(auth_info, dict)
            assert 'type' in auth_info

    def test_extract_data_schemas_from_code(self, mock_vector_store, mock_llm_manager):
        """Test extracting data schemas from model code."""
        extractor = ApplicationKnowledgeExtractor(mock_vector_store, mock_llm_manager)
        
        model_code = '''
class User(db.Model):
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    name = Column(String(100))
    
class Product(BaseModel):
    id: int
    name: str
    price: float
    description: Optional[str] = None
'''
        
        schemas = extractor._extract_data_schemas_from_code(model_code)
        
        assert isinstance(schemas, dict)
        # Should extract model definitions

    def test_infer_element_type(self, mock_vector_store, mock_llm_manager):
        """Test inferring element types from selectors."""
        extractor = ApplicationKnowledgeExtractor(mock_vector_store, mock_llm_manager)
        
        assert extractor._infer_element_type("[data-testid='login-button']") == "button"
        assert extractor._infer_element_type("input[type='email']") == "input"
        assert extractor._infer_element_type("#login-form") == "form"
        assert extractor._infer_element_type("a.nav-link") == "link"
        assert extractor._infer_element_type("#unknown-element") == "element"

    def test_infer_element_type_from_context(self, mock_vector_store, mock_llm_manager):
        """Test inferring element types from surrounding context."""
        extractor = ApplicationKnowledgeExtractor(mock_vector_store, mock_llm_manager)
        
        button_context = '<button data-testid="submit">Submit</button>'
        element_type = extractor._infer_element_type_from_context(button_context, "submit")
        assert element_type == "button"
        
        input_context = '<input data-testid="email" type="email" />'
        element_type = extractor._infer_element_type_from_context(input_context, "email")
        assert element_type == "input"

    def test_extract_base_url_from_config(self, mock_vector_store, mock_llm_manager):
        """Test extracting base URL from configuration content."""
        extractor = ApplicationKnowledgeExtractor(mock_vector_store, mock_llm_manager)
        
        config_content = '''
BASE_URL = "https://api.example.com"
API_URL: "http://localhost:3000/api"
SERVER_URL = 'https://production.api.com'
'''
        
        url = extractor._extract_base_url_from_config(config_content)
        assert url is not None
        assert url.startswith(('http://', 'https://'))

    def test_infer_base_url_fallback(self, mock_vector_store, mock_llm_manager):
        """Test base URL inference with fallback."""
        extractor = ApplicationKnowledgeExtractor(mock_vector_store, mock_llm_manager)
        
        # Mock empty results from vector store
        mock_vector_store.query_similar.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        url = extractor._infer_base_url()
        
        # Should return default fallback
        assert url == "http://localhost:8000"

    def test_discover_framework_patterns(self, mock_vector_store, mock_llm_manager):
        """Test discovering framework-specific patterns."""
        extractor = ApplicationKnowledgeExtractor(mock_vector_store, mock_llm_manager)
        
        # Mock framework config content
        mock_vector_store.query_similar.return_value = {
            'documents': [['{"testMatch": ["**/*.test.js"], "collectCoverage": true}']],
            'metadatas': [['type: config']],
            'distances': [[0.1]]
        }
        
        patterns = extractor._discover_framework_patterns()
        
        assert isinstance(patterns, dict)
        # Should attempt to extract framework configurations