"""Tests for the TestTeller automation functionality."""

import pytest

pytestmark = pytest.mark.automation
import tempfile
from pathlib import Path

from testteller.automator_agent.parser.markdown_parser import MarkdownTestCaseParser, TestCase
from testteller.automator_agent.rag_enhanced_generator import RAGEnhancedTestGenerator
from testteller.automator_agent.base_generator import BaseTestGenerator


class TestMarkdownParser:
    """Test markdown test case parser."""
    
    def test_parse_simple_e2e_test(self):
        """Test parsing a simple E2E test case."""
        markdown_content = """
### Test Case E2E_[1]
**Feature:** User Login
**Type:** Authentication Flow
**Category:** Happy Path

#### Objective
Verify that users can successfully log in with valid credentials.

#### References
- **Product:** Login Feature PRD
- **Technical:** Authentication API

#### Prerequisites & Setup
- **System State:** User account exists
- **Test Data:** user_id: 123, email: test@example.com

#### Test Steps
1. **Action:** Navigate to login page
   - **Technical Details:** GET request to /login
2. **Validation:** Verify login form is displayed
   - **Technical Details:** Check for username and password fields

#### Expected Final State
- **UI/Frontend:** Dashboard page is displayed
- **Backend/API:** User session is created
"""
        
        parser = MarkdownTestCaseParser()
        test_cases = parser.parse_content(markdown_content)
        
        assert len(test_cases) == 1
        test_case = test_cases[0]
        
        assert test_case.id == "E2E_[1]"
        assert test_case.feature == "User Login"
        assert test_case.type == "Authentication Flow"
        assert test_case.category == "Happy Path"
        assert "successfully log in" in test_case.objective
        assert len(test_case.test_steps) == 2
        assert test_case.prerequisites
        assert test_case.expected_state


class TestRAGEnhancedGenerator:
    """Test RAG Enhanced test code generator."""
    
    def test_generate_python_pytest_tests(self):
        """Test generating Python pytest tests."""
        test_case = TestCase(
            id="E2E_[1]",
            feature="User Login",
            type="Authentication",
            category="Happy Path",
            objective="Test user login functionality"
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            # Mock the required parameters for RAGEnhancedTestGenerator
            # Constructor signature: __init__(self, framework, output_dir, vector_store, language='python', llm_manager=None, num_context_docs=5)
            from unittest.mock import Mock
            mock_vector_store = Mock()
            
            generator = RAGEnhancedTestGenerator(
                framework="pytest",
                output_dir=output_dir,
                vector_store=mock_vector_store,
                language="python",
                llm_manager=None,
                num_context_docs=5
            )
            
            # Mock the generate method since it requires external dependencies
            # This test focuses on the class instantiation and basic structure
            assert generator.language == "python"
            assert generator.framework == "pytest"
            assert generator.output_dir == output_dir


class TestJavaScriptGenerator:
    """Test JavaScript test code generator."""
    
    def test_generate_javascript_jest_tests(self):
        """Test generating JavaScript Jest tests."""
        test_case = TestCase(
            id="INT_[1]",
            feature="API Integration",
            type="API",
            category="Contract",
            objective="Test API integration functionality"
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            # Mock the required parameters for RAGEnhancedTestGenerator
            # Constructor signature: __init__(self, framework, output_dir, vector_store, language='python', llm_manager=None, num_context_docs=5)
            from unittest.mock import Mock
            mock_vector_store = Mock()
            
            generator = RAGEnhancedTestGenerator(
                framework="jest",
                output_dir=output_dir,
                vector_store=mock_vector_store,
                language="javascript",
                llm_manager=None,
                num_context_docs=5
            )
            
            # Mock the generate method since it requires external dependencies
            # This test focuses on the class instantiation and basic structure
            assert generator.language == "javascript"
            assert generator.framework == "jest"
            assert generator.output_dir == output_dir


class TestGeneratorUtils:
    """Test generator utility functions."""
    
    def test_base_generator_initialization(self):
        """Test base generator initialization."""
        # Create a simple mock class to test abstract BaseTestGenerator
        class MockGenerator(BaseTestGenerator):
            def generate(self, test_cases):
                return {}
            
            def get_supported_frameworks(self):
                return ["pytest", "unittest"]
            
            def get_file_extension(self):
                return ".py"
        
        generator = MockGenerator("pytest", Path("."))
        
        assert generator.framework == "pytest"
        assert generator.output_dir == Path(".")
        assert generator.get_file_extension() == ".py"
        assert "pytest" in generator.get_supported_frameworks()
    
    def test_test_case_creation(self):
        """Test TestCase data class creation."""
        test_case = TestCase(
            id="E2E_[1]",
            feature="User Login",
            type="Authentication",
            category="Happy Path",
            objective="Test login functionality"
        )
        
        assert test_case.id == "E2E_[1]"
        assert test_case.feature == "User Login"
        assert test_case.type == "Authentication"
        assert test_case.category == "Happy Path"
        assert test_case.objective == "Test login functionality"