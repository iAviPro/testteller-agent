"""Unit tests for RAG-Enhanced Test Generator."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List

from testteller.automator_agent.rag_enhanced_generator import (
    RAGEnhancedTestGenerator,
    TestCodeValidator,
    ValidationResult
)
from testteller.automator_agent.application_context import (
    ApplicationContext,
    ApplicationKnowledgeExtractor,
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
    mock_vs.query_similar.return_value = {
        'documents': [['sample code content', 'API endpoint definition']],
        'metadatas': [['file_type: .py', 'type: code']],
        'distances': [[0.1, 0.2]]
    }
    return mock_vs


@pytest.fixture
def mock_llm_manager():
    """Mock LLM manager for testing."""
    mock_llm = Mock(spec=LLMManager)
    mock_llm.generate_text.return_value = """import pytest
import requests

def test_user_login():
    response = requests.post('/api/auth/login', json={'email': 'test@test.com', 'password': 'password'})
    assert response.status_code == 200
    assert 'token' in response.json()
"""
    return mock_llm


@pytest.fixture
def sample_test_cases():
    """Sample test cases for testing."""
    return [
        TestCase(
            id="E2E_[1]",
            feature="User Authentication",
            type="E2E",
            category="Authentication",
            objective="Test user login functionality",
            test_steps=[
                TestStep(action="Navigate to login page", validation="Login form is displayed"),
                TestStep(action="Enter valid credentials", validation="User is logged in")
            ]
        ),
        TestCase(
            id="API_[1]",
            feature="User API",
            type="API",
            category="Integration",
            objective="Test user API endpoints",
            test_steps=[
                TestStep(action="Send GET request to /api/users", validation="Returns user list")
            ]
        )
    ]


@pytest.fixture
def sample_app_context():
    """Sample application context for testing."""
    return ApplicationContext(
        base_url="https://api.example.com",
        api_endpoints={
            "GET:/api/users": APIEndpoint("/api/users", "GET", "Get users"),
            "POST:/api/auth/login": APIEndpoint("/api/auth/login", "POST", "User login", auth_required=False)
        },
        ui_selectors={
            "[data-testid='login-email']": UIPattern("[data-testid='login-email']", "input", "Email input field"),
            "[data-testid='login-password']": UIPattern("[data-testid='login-password']", "input", "Password input field"),
            "[data-testid='login-submit']": UIPattern("[data-testid='login-submit']", "button", "Login submit button")
        },
        auth_patterns=AuthPattern(
            auth_type="jwt",
            login_endpoint="/api/auth/login",
            token_header="Authorization",
            login_selectors={
                "email": "[data-testid='login-email']",
                "password": "[data-testid='login-password']"
            }
        ),
        data_schemas={
            "User": DataSchema("User", {"id": "int", "email": "str", "name": "str"}, ["id", "email"])
        }
    )


class TestRAGEnhancedTestGenerator:
    """Test RAG-Enhanced Test Generator functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self, mock_vector_store, mock_llm_manager):
        """Test RAG generator initialization."""
        generator = RAGEnhancedTestGenerator(
            framework="pytest",
            output_dir=self.temp_dir,
            vector_store=mock_vector_store,
            language="python",
            llm_manager=mock_llm_manager,
            num_context_docs=5
        )

        assert generator.framework == "pytest"
        assert generator.language == "python"
        assert generator.num_context_docs == 5
        assert generator.vector_store == mock_vector_store
        assert generator.llm_manager == mock_llm_manager
        assert isinstance(generator.knowledge_extractor, ApplicationKnowledgeExtractor)
        assert isinstance(generator.validator, TestCodeValidator)

    def test_get_supported_frameworks(self, mock_vector_store, mock_llm_manager):
        """Test getting supported frameworks for different languages."""
        generator = RAGEnhancedTestGenerator(
            framework="pytest",
            output_dir=self.temp_dir,
            vector_store=mock_vector_store,
            language="python",
            llm_manager=mock_llm_manager
        )

        python_frameworks = generator.get_supported_frameworks()
        assert "pytest" in python_frameworks
        assert "unittest" in python_frameworks
        assert "playwright" in python_frameworks

        # Test JavaScript frameworks
        generator.language = "javascript"
        js_frameworks = generator.get_supported_frameworks()
        assert "jest" in js_frameworks
        assert "mocha" in js_frameworks
        assert "playwright" in js_frameworks

    def test_get_file_extension(self, mock_vector_store, mock_llm_manager):
        """Test getting file extensions for different languages."""
        generator = RAGEnhancedTestGenerator(
            framework="pytest",
            output_dir=self.temp_dir,
            vector_store=mock_vector_store,
            language="python",
            llm_manager=mock_llm_manager
        )

        assert generator.get_file_extension() == ".py"

        generator.language = "javascript"
        assert generator.get_file_extension() == ".js"

        generator.language = "typescript"
        assert generator.get_file_extension() == ".ts"

    @pytest.mark.asyncio
    @patch('testteller.automator_agent.rag_enhanced_generator.ApplicationKnowledgeExtractor')
    async def test_generate_with_app_context(self, mock_extractor, mock_vector_store, mock_llm_manager, 
                                     sample_test_cases, sample_app_context):
        """Test code generation with application context."""
        # Mock knowledge extractor
        mock_extractor_instance = Mock()
        mock_extractor.return_value = mock_extractor_instance
        mock_extractor_instance.extract_app_context.return_value = sample_app_context

        generator = RAGEnhancedTestGenerator(
            framework="pytest",
            output_dir=self.temp_dir,
            vector_store=mock_vector_store,
            language="python",
            llm_manager=mock_llm_manager
        )

        # Mock the validator to return valid code
        with patch.object(generator.validator, 'validate_generated_test') as mock_validate:
            mock_validate.return_value = ValidationResult(
                is_valid=True, issues=[], confidence_score=0.9
            )

            result = await generator.generate(sample_test_cases)

            # Verify results
            assert isinstance(result, dict)
            assert len(result) > 0
            
            # Check that files are generated for different categories
            generated_files = list(result.keys())
            assert any("test_" in filename for filename in generated_files)
            
            # Verify knowledge extractor was called
            mock_extractor_instance.extract_app_context.assert_called_once_with(sample_test_cases)

    def test_categorize_tests(self, mock_vector_store, mock_llm_manager, sample_test_cases):
        """Test test case categorization."""
        generator = RAGEnhancedTestGenerator(
            framework="pytest",
            output_dir=self.temp_dir,
            vector_store=mock_vector_store,
            language="python",
            llm_manager=mock_llm_manager
        )

        categorized = generator.categorize_tests(sample_test_cases)

        assert isinstance(categorized, dict)
        # Should have at least E2E and API categories based on sample test cases
        assert len(categorized) >= 2

    @pytest.mark.asyncio
    @patch('testteller.automator_agent.rag_enhanced_generator.ApplicationKnowledgeExtractor')
    async def test_generate_with_validation_errors(self, mock_extractor, mock_vector_store, 
                                           mock_llm_manager, sample_test_cases, sample_app_context):
        """Test code generation with validation errors that get fixed."""
        # Mock knowledge extractor
        mock_extractor_instance = Mock()
        mock_extractor.return_value = mock_extractor_instance
        mock_extractor_instance.extract_app_context.return_value = sample_app_context

        generator = RAGEnhancedTestGenerator(
            framework="pytest",
            output_dir=self.temp_dir,
            vector_store=mock_vector_store,
            language="python",
            llm_manager=mock_llm_manager
        )

        # Mock validator to first return errors, then success after fix
        validation_results = [
            ValidationResult(is_valid=False, issues=["Missing imports", "TODO placeholders"], confidence_score=0.3),
            ValidationResult(is_valid=True, issues=[], confidence_score=0.9)
        ]
        
        with patch.object(generator.validator, 'validate_generated_test') as mock_validate, \
             patch.object(generator.validator, 'fix_validation_issues') as mock_fix:
            
            mock_validate.side_effect = validation_results
            mock_fix.return_value = "# Fixed code"

            result = await generator.generate(sample_test_cases)

            # Verify validation and fixing were attempted
            assert mock_validate.call_count >= 1
            assert mock_fix.call_count >= 1
            assert isinstance(result, dict)

    def test_generate_supporting_files(self, mock_vector_store, mock_llm_manager, sample_app_context):
        """Test generation of supporting files."""
        generator = RAGEnhancedTestGenerator(
            framework="pytest",
            output_dir=self.temp_dir,
            vector_store=mock_vector_store,
            language="python",
            llm_manager=mock_llm_manager
        )

        supporting_files = generator._generate_supporting_files(sample_app_context)

        assert isinstance(supporting_files, dict)
        # Should generate requirements.txt for Python
        assert "requirements.txt" in supporting_files
        
        # For pytest, should also generate conftest.py
        if generator.framework == "pytest":
            assert "conftest.py" in supporting_files
            conftest_content = supporting_files["conftest.py"]
            assert "pytest.fixture" in conftest_content
            assert sample_app_context.base_url in conftest_content


class TestTestCodeValidator:
    """Test code validation functionality."""

    def test_initialization(self, mock_vector_store, mock_llm_manager):
        """Test validator initialization."""
        validator = TestCodeValidator(mock_vector_store, mock_llm_manager)
        
        assert validator.vector_store == mock_vector_store
        assert validator.llm_manager == mock_llm_manager

    def test_validate_valid_python_code(self, mock_vector_store, mock_llm_manager):
        """Test validation of valid Python test code."""
        validator = TestCodeValidator(mock_vector_store, mock_llm_manager)
        
        valid_code = """
import pytest
import requests

def test_user_login():
    response = requests.post('/api/login', json={'email': 'test@test.com'})
    assert response.status_code == 200
"""
        
        result = validator.validate_generated_test(valid_code, "python")
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid == True
        assert len(result.issues) == 0
        assert result.confidence_score > 0.5

    def test_validate_invalid_python_code(self, mock_vector_store, mock_llm_manager):
        """Test validation of invalid Python test code."""
        validator = TestCodeValidator(mock_vector_store, mock_llm_manager)
        
        invalid_code = """
# No imports

def test_something():
    # TODO: Implement this test
    pass
"""
        
        result = validator.validate_generated_test(invalid_code, "python")
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid == False
        assert len(result.issues) > 0
        assert "TODO" in str(result.issues) or "Missing required imports" in str(result.issues)

    def test_fix_validation_issues(self, mock_vector_store, mock_llm_manager, sample_app_context):
        """Test fixing validation issues."""
        validator = TestCodeValidator(mock_vector_store, mock_llm_manager)
        
        # Mock LLM to return fixed code
        mock_llm_manager.generate_text.return_value = """
import pytest
import requests

def test_login():
    response = requests.post('/api/login', json={'email': 'test@test.com'})
    assert response.status_code == 200
"""
        
        problematic_code = """
def test_login():
    # TODO: Implement login test
    pass
"""
        
        issues = ["Contains TODO/FIXME placeholders", "Missing required imports"]
        
        fixed_code = validator.fix_validation_issues(problematic_code, issues, sample_app_context)
        
        assert isinstance(fixed_code, str)
        assert len(fixed_code) > len(problematic_code)
        mock_llm_manager.generate_text.assert_called_once()


class TestValidationResult:
    """Test validation result data structure."""

    def test_validation_result_creation(self):
        """Test creating validation results."""
        result = ValidationResult(
            is_valid=True,
            issues=[],
            confidence_score=0.95
        )
        
        assert result.is_valid == True
        assert result.issues == []
        assert result.confidence_score == 0.95
        
        # Test with issues
        result_with_issues = ValidationResult(
            is_valid=False,
            issues=["Syntax error", "Missing imports"],
            confidence_score=0.3
        )
        
        assert result_with_issues.is_valid == False
        assert len(result_with_issues.issues) == 2
        assert result_with_issues.confidence_score == 0.3