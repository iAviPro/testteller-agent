"""Integration tests for TestTeller RAG-enhanced automation functionality."""

import pytest
import tempfile
from pathlib import Path
import os
from unittest.mock import Mock, patch, AsyncMock

from testteller.automator_agent.parser.markdown_parser import MarkdownTestCaseParser
from testteller.automator_agent.rag_enhanced_generator import RAGEnhancedTestGenerator
from testteller.automator_agent.application_context import ApplicationKnowledgeExtractor, ApplicationContext
from testteller.automator_agent.cli import automate_command
from testteller.core.vector_store.chromadb_manager import ChromaDBManager
from testteller.core.llm.llm_manager import LLMManager

pytestmark = pytest.mark.automation


@pytest.fixture
def mock_vector_store():
    """Mock vector store for integration tests."""
    mock_vs = Mock(spec=ChromaDBManager)
    mock_vs.query_similar.return_value = {
        'documents': [[
            'def test_existing_login(): pass',
            'page.click("[data-testid=\'login-btn\']")',
            'requests.post("/api/auth/login")',
            'class User(db.Model): email = Column(String)'
        ]],
        'metadatas': [['type: code'] * 4],
        'distances': [[0.1, 0.2, 0.3, 0.4]]
    }
    mock_vs.list_collections.return_value = ['test_collection']
    return mock_vs


@pytest.fixture 
def mock_llm_manager():
    """Mock LLM manager for integration tests."""
    mock_llm = Mock(spec=LLMManager)
    mock_llm.provider = "gemini"
    mock_llm.generate_text.return_value = """
import pytest
import requests
from playwright.sync_api import Page, expect

@pytest.fixture
def base_url():
    return "https://api.example.com"

def test_user_login_success(page: Page, base_url):
    \"\"\"Test successful user login.\"\"\"
    page.goto(f"{base_url}/login")
    page.fill("[data-testid='email-input']", "test@example.com")
    page.fill("[data-testid='password-input']", "password123")
    page.click("[data-testid='login-submit']")
    expect(page.locator("[data-testid='dashboard']")).to_be_visible()
    
def test_api_get_users(base_url):
    \"\"\"Test GET /api/users endpoint.\"\"\"
    response = requests.get(f"{base_url}/api/users", 
                          headers={"Authorization": "Bearer test_token"})
    assert response.status_code == 200
    assert isinstance(response.json(), list)
"""
    return mock_llm


class TestRAGEnhancedEndToEndAutomation:
    """Test complete RAG-enhanced automation workflow from markdown to generated code."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create comprehensive test cases markdown
        self.test_cases_content = """
# Generated Test Cases

## Documentation Analysis
- **Available Documentation Types:** Product Requirements, API Contracts, Technical Design
- **Missing Documentation Types:** Database Schemas, Event Specifications

## Test Case Distribution
- **End-to-End Tests:** 2 test cases
- **Integration Tests:** 2 test cases
- **Technical Tests:** 1 test case

## Coverage Analysis
- **Scenario Distribution:** 40% negative/edge cases, 30% happy paths, 15% failure scenarios, 15% non-functional
- **Technical Depth:** High coverage of API contracts and system architecture

---

### Test Case E2E_[1]
**Feature:** User Authentication System
**Type:** End-to-End
**Category:** Happy Path

#### Objective
Verify that users can successfully log in with valid credentials and access the main dashboard.

#### Test Steps
1. **Action:** Navigate to the login page
   - **Technical Details:** Open browser and navigate to `/login` endpoint
2. **Action:** Enter valid email address in the email field
   - **Technical Details:** Use test email `test@example.com`
3. **Action:** Enter valid password in the password field
   - **Technical Details:** Use test password `password123`
4. **Action:** Click the "Login" button
   - **Technical Details:** Submit the login form via button click
5. **Validation:** User is redirected to the dashboard
   - **Technical Details:** Verify URL contains `/dashboard` and dashboard content loads

### Test Case API_[1]
**Feature:** User Management API
**Type:** API
**Category:** CRUD Operations

#### Objective
Verify that the user management API endpoints function correctly for retrieving user data.

#### Test Steps
1. **Action:** Send authenticated GET request to `/api/users` endpoint
   - **Technical Details:** Include valid JWT token in Authorization header
2. **Validation:** Response contains list of users
   - **Technical Details:** Status code 200, Content-Type application/json, response body is array
3. **Action:** Send GET request to `/api/users/123` endpoint
   - **Technical Details:** Request specific user by ID with authentication
4. **Validation:** Response contains user details
   - **Technical Details:** Status code 200, response contains user object with id, email, name fields

### Test Case INT_[1]
**Feature:** Order Processing Integration
**Type:** Integration
**Category:** Service Communication

#### Objective
Verify that the User Service correctly communicates with the Order Service for order processing.

#### Test Steps
1. **Action:** User Service sends order creation request to Order Service
   - **Technical Details:** POST to Order Service with user context and order details
2. **Validation:** Order Service acknowledges receipt
   - **Technical Details:** Returns order ID and confirmation status
3. **Action:** User Service updates user's order history
   - **Technical Details:** Store order reference in user profile
4. **Validation:** User profile reflects new order
   - **Technical Details:** GET /api/users/{id}/orders returns the new order
"""
        
        self.test_file = self.temp_dir / "test_cases.md"
        with open(self.test_file, 'w') as f:
            f.write(self.test_cases_content)
            
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_markdown_parser_with_rag_content(self):
        """Test that markdown parser can extract test cases for RAG enhancement."""
        parser = MarkdownTestCaseParser()
        test_cases = parser.parse_file(self.test_file)
        
        assert len(test_cases) >= 3
        
        # Check E2E test case
        e2e_case = next((tc for tc in test_cases if tc.id == "E2E_[1]"), None)
        assert e2e_case is not None
        assert e2e_case.feature == "User Authentication System"
        assert e2e_case.type == "End-to-End"
        assert e2e_case.category == "Happy Path"
        assert len(e2e_case.test_steps) >= 4
        
        # Check API test case
        api_case = next((tc for tc in test_cases if tc.id == "API_[1]"), None)
        assert api_case is not None
        assert api_case.feature == "User Management API"
        assert api_case.type == "API"
        
        # Check Integration test case
        int_case = next((tc for tc in test_cases if tc.id == "INT_[1]"), None)
        assert int_case is not None
        assert int_case.feature == "Order Processing Integration"
        assert int_case.type == "Integration"

    def test_rag_enhanced_generator_integration(self, mock_vector_store, mock_llm_manager):
        """Test RAG-enhanced generator with real-like workflow."""
        parser = MarkdownTestCaseParser()
        test_cases = parser.parse_file(self.test_file)
        
        output_dir = self.temp_dir / "generated"
        
        generator = RAGEnhancedTestGenerator(
            framework="playwright",
            output_dir=output_dir,
            vector_store=mock_vector_store,
            language="python",
            llm_manager=mock_llm_manager,
            num_context_docs=5
        )
        
        # Mock the validator to always pass
        generator.validator.validate_generated_test.return_value = Mock(
            is_valid=True, issues=[], confidence_score=0.9
        )
        
        generated_files = generator.generate(test_cases)
        
        # Verify files were generated
        assert isinstance(generated_files, dict)
        assert len(generated_files) > 0
        
        # Verify vector store was queried for context
        assert mock_vector_store.query_similar.call_count >= 3
        
        # Verify LLM was called for generation
        assert mock_llm_manager.generate_text.call_count >= 1
        
        # Check generated file extensions
        for filename in generated_files.keys():
            if filename.endswith('.py'):
                assert 'test_' in filename
                
        # Verify supporting files
        assert any(filename.endswith('.txt') for filename in generated_files.keys())  # requirements.txt

    @patch('testteller.automator_agent.cli.initialize_vector_store')
    @patch('testteller.automator_agent.cli.LLMManager')
    @patch('testteller.automator_agent.cli.UnifiedDocumentParser')
    @patch('testteller.automator_agent.cli.RAGEnhancedTestGenerator')
    def test_complete_cli_workflow(self, mock_rag_gen, mock_parser, mock_llm, mock_init_vs):
        """Test complete CLI workflow with RAG enhancement."""
        output_dir = self.temp_dir / "cli_output"
        
        # Mock vector store initialization
        mock_vs = Mock()
        mock_init_vs.return_value = mock_vs
        
        # Mock LLM manager
        mock_llm_instance = Mock()
        mock_llm_instance.provider = "claude"
        mock_llm.return_value = mock_llm_instance
        
        # Mock unified parser
        mock_parsed_doc = Mock()
        mock_parsed_doc.test_cases = [
            Mock(id="E2E_1", feature="Login", type="E2E", category="Auth", 
                 objective="Test login", test_steps=[])
        ]
        mock_parsed_doc.metadata = Mock(title="Test Cases", word_count=500)
        
        mock_parser_instance = Mock()
        mock_parser.return_value = mock_parser_instance
        mock_parser_instance.parse_for_automation = AsyncMock(return_value=mock_parsed_doc)
        
        # Mock RAG generator
        mock_generator = Mock()
        mock_rag_gen.return_value = mock_generator
        mock_generator.generate.return_value = {
            "test_e2e.py": "# Generated E2E test with real selectors",
            "requirements.txt": "pytest>=7.0.0\nplaywright>=1.40.0"
        }
        mock_generator.write_files.return_value = None
        
        # Run the CLI command
        automate_command(
            input_file=str(self.test_file),
            collection_name="integration_test_collection",
            language="python",
            framework="playwright",
            output_dir=str(output_dir),
            interactive=False,
            num_context_docs=8,
            verbose=True
        )
        
        # Verify the workflow
        mock_init_vs.assert_called_once()
        mock_llm.assert_called_once()
        mock_parser_instance.parse_for_automation.assert_called_once()
        mock_rag_gen.assert_called_once()
        mock_generator.generate.assert_called_once()
        mock_generator.write_files.assert_called_once()

    def test_application_context_extraction_integration(self, mock_vector_store, mock_llm_manager):
        """Test application context extraction in integration scenario."""
        parser = MarkdownTestCaseParser()
        test_cases = parser.parse_file(self.test_file)
        
        # Create knowledge extractor
        extractor = ApplicationKnowledgeExtractor(
            mock_vector_store, mock_llm_manager, num_context_docs=10
        )
        
        # Mock vector store to return realistic content for different queries
        mock_vector_store.query_similar.side_effect = [
            # API endpoints query
            {
                'documents': [['@app.route("/api/users", methods=["GET"])', 
                              '@app.route("/api/auth/login", methods=["POST"])']],
                'metadatas': [['type: code'] * 2],
                'distances': [[0.1, 0.2]]
            },
            # OpenAPI query  
            {
                'documents': [['{"paths": {"/api/users": {"get": {"summary": "Get users"}}}}']],
                'metadatas': [['type: documentation']],
                'distances': [[0.15]]
            },
            # UI test patterns query
            {
                'documents': [['page.click("[data-testid=\'login-btn\']")',
                              'page.fill("[data-testid=\'email\']", email)']],
                'metadatas': [['type: code'] * 2],
                'distances': [[0.1, 0.2]]
            },
            # UI component query
            {
                'documents': [['<button data-testid="login-btn">Login</button>',
                              '<input data-testid="email" type="email" />']],
                'metadatas': [['type: code'] * 2],
                'distances': [[0.1, 0.2]]
            },
            # Auth patterns query
            {
                'documents': [['JWT token authentication', 'Bearer token required']],
                'metadatas': [['type: code'] * 2],
                'distances': [[0.1, 0.2]]
            },
            # Data schemas query
            {
                'documents': [['class User(db.Model): id = Column(Integer)']],
                'metadatas': [['type: code']],
                'distances': [[0.1]]
            },
            # Test patterns query
            {
                'documents': [['def test_login(): page.goto("/login")']],
                'metadatas': [['type: code']],
                'distances': [[0.1]]
            },
            # Framework config query
            {
                'documents': [['{"testMatch": ["**/*.test.js"]}']],
                'metadatas': [['type: config']],
                'distances': [[0.1]]
            },
            # Base URL query
            {
                'documents': [['BASE_URL = "https://api.example.com"']],
                'metadatas': [['type: config']],
                'distances': [[0.1]]
            }
        ]
        
        app_context = extractor.extract_app_context(test_cases)
        
        # Verify context extraction
        assert isinstance(app_context, ApplicationContext)
        assert mock_vector_store.query_similar.call_count >= 7  # Multiple discovery queries
        
        # Context should have realistic structure
        assert app_context.base_url is not None  # Should have inferred or default URL

    def test_error_handling_and_fallbacks(self, mock_vector_store, mock_llm_manager):
        """Test error handling and fallback mechanisms in integration."""
        parser = MarkdownTestCaseParser()
        test_cases = parser.parse_file(self.test_file)
        
        # Simulate vector store failure
        mock_vector_store.query_similar.side_effect = Exception("Vector store connection failed")
        
        output_dir = self.temp_dir / "fallback_output"
        
        generator = RAGEnhancedTestGenerator(
            framework="pytest",
            output_dir=output_dir,
            vector_store=mock_vector_store,
            language="python",
            llm_manager=mock_llm_manager
        )
        
        # Should still generate files using fallback mechanism
        generated_files = generator.generate(test_cases)
        
        assert isinstance(generated_files, dict)
        assert len(generated_files) > 0  # Should have fallback files
        
        # Verify fallback files contain basic structure
        for filename, content in generated_files.items():
            if filename.endswith('.py'):
                assert isinstance(content, str)
                assert len(content) > 0

    def test_multi_language_generation_integration(self, mock_vector_store, mock_llm_manager):
        """Test generation for multiple programming languages."""
        parser = MarkdownTestCaseParser()
        test_cases = parser.parse_file(self.test_file)
        
        languages_frameworks = [
            ("python", "pytest"),
            ("javascript", "jest"),
            ("python", "playwright")
        ]
        
        for language, framework in languages_frameworks:
            output_dir = self.temp_dir / f"{language}_{framework}_output"
            
            # Adjust LLM response based on language/framework
            if language == "javascript":
                mock_llm_manager.generate_text.return_value = """
const { test, expect } = require('@playwright/test');

test('user login success', async ({ page }) => {
  await page.goto('/login');
  await page.fill('[data-testid="email-input"]', 'test@example.com');
  await page.fill('[data-testid="password-input"]', 'password123');
  await page.click('[data-testid="login-submit"]');
  await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
});
"""
            
            generator = RAGEnhancedTestGenerator(
                framework=framework,
                output_dir=output_dir,
                vector_store=mock_vector_store,
                language=language,
                llm_manager=mock_llm_manager
            )
            
            # Mock validator
            generator.validator.validate_generated_test.return_value = Mock(
                is_valid=True, issues=[], confidence_score=0.8
            )
            
            generated_files = generator.generate(test_cases)
            
            assert isinstance(generated_files, dict)
            assert len(generated_files) > 0
            
            # Verify correct file extensions
            expected_extension = generator.get_file_extension()
            test_files = [f for f in generated_files.keys() if f.startswith('test_')]
            assert any(f.endswith(expected_extension) for f in test_files)


class TestRAGEnhancedQualityAssurance:
    """Test quality assurance aspects of RAG-enhanced automation."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generated_code_quality_metrics(self, mock_vector_store, mock_llm_manager):
        """Test quality metrics of generated code."""
        # Create a simple test case
        test_cases = [
            Mock(id="TEST_1", feature="Login", type="E2E", category="Auth",
                 objective="Test login", test_steps=[])
        ]
        
        # Mock high-quality generation
        mock_llm_manager.generate_text.return_value = """
import pytest
import requests
from playwright.sync_api import Page, expect

@pytest.fixture
def base_url():
    return "https://api.example.com"

def test_user_login_complete():
    \"\"\"Test complete user login flow with validation.\"\"\"
    # Setup test data
    test_email = "test@example.com" 
    test_password = "secure_password123"
    
    # Navigate to login page
    page.goto(f"{base_url}/login")
    
    # Fill login form
    page.fill("[data-testid='email-input']", test_email)
    page.fill("[data-testid='password-input']", test_password)
    
    # Submit form and verify success
    page.click("[data-testid='login-submit']")
    expect(page.locator("[data-testid='dashboard']")).to_be_visible()
    expect(page.locator("[data-testid='user-menu']")).to_contain_text("test@example.com")
"""
        
        generator = RAGEnhancedTestGenerator(
            framework="playwright",
            output_dir=self.temp_dir,
            vector_store=mock_vector_store,
            language="python",
            llm_manager=mock_llm_manager
        )
        
        # Mock validator to return high quality
        generator.validator.validate_generated_test.return_value = Mock(
            is_valid=True, issues=[], confidence_score=0.95
        )
        
        generated_files = generator.generate(test_cases)
        
        # Import the quality assessment function and test it
        from testteller.automator_agent.cli import assess_generated_quality
        
        # Capture the quality assessment output
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        assess_generated_quality(generated_files)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Verify quality metrics are calculated
        assert "Quality Assessment:" in output
        assert "Total Lines of Code:" in output
        assert "Test Functions:" in output
        assert "Quality Score:" in output
        
        # Should have high quality score (no TODOs in the mock)
        assert "TODO Items: 0" in output or "TODO Items: " in output

    def test_validation_and_fixing_integration(self, mock_vector_store, mock_llm_manager):
        """Test validation and automatic fixing of generated code."""
        test_cases = [
            Mock(id="TEST_1", feature="API", type="API", category="CRUD",
                 objective="Test API", test_steps=[])
        ]
        
        # Mock LLM to first generate problematic code, then fixed code
        mock_llm_manager.generate_text.side_effect = [
            # Initial generation with issues
            """
def test_api_endpoint():
    # TODO: Implement API test
    pass
""",
            # Fixed generation
            """
import requests
import pytest

def test_api_endpoint():
    \"\"\"Test API endpoint functionality.\"\"\"
    response = requests.get("/api/users")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
"""
        ]
        
        generator = RAGEnhancedTestGenerator(
            framework="pytest",
            output_dir=self.temp_dir,
            vector_store=mock_vector_store,
            language="python",
            llm_manager=mock_llm_manager
        )
        
        # Mock validator to first fail, then pass
        generator.validator.validate_generated_test.side_effect = [
            Mock(is_valid=False, issues=["Contains TODO/FIXME placeholders"], confidence_score=0.2),
            Mock(is_valid=True, issues=[], confidence_score=0.9)
        ]
        
        generator.validator.fix_validation_issues.return_value = """
import requests
import pytest

def test_api_endpoint():
    \"\"\"Test API endpoint functionality.\"\"\"
    response = requests.get("/api/users")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
"""
        
        generated_files = generator.generate(test_cases)
        
        # Verify validation and fixing workflow
        assert generator.validator.validate_generated_test.call_count >= 1
        assert generator.validator.fix_validation_issues.call_count >= 1
        
        # Verify final files are better quality
        assert isinstance(generated_files, dict)
        for content in generated_files.values():
            if content.startswith('import') or content.startswith('def'):
                assert 'TODO' not in content  # Should be fixed