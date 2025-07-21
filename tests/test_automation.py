"""Tests for the TestWriter automation functionality."""

import pytest

pytestmark = pytest.mark.automation
import tempfile
from pathlib import Path

from testteller.automator_agent.parser.markdown_parser import MarkdownTestCaseParser, TestCase
from testteller.automator_agent.generators.python_generator import PythonTestGenerator
from testteller.automator_agent.generators.javascript_generator import JavaScriptTestGenerator


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


class TestPythonGenerator:
    """Test Python test code generator."""
    
    def test_generate_pytest_tests(self):
        """Test generating pytest tests."""
        test_case = TestCase(
            id="E2E_[1]",
            feature="User Login",
            type="Authentication",
            category="Happy Path",
            objective="Test user login functionality"
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            generator = PythonTestGenerator("pytest", output_dir)
            
            generated_files = generator.generate([test_case])
            
            assert "test_e2e.py" in generated_files
            assert "conftest.py" in generated_files
            assert "requirements.txt" in generated_files
            
            # Check that pytest imports are present
            test_content = generated_files["test_e2e.py"]
            assert "import pytest" in test_content
            assert "def test_e2e_1" in test_content


class TestJavaScriptGenerator:
    """Test JavaScript test code generator."""
    
    def test_generate_jest_tests(self):
        """Test generating Jest tests."""
        test_case = TestCase(
            id="INT_[1]",
            feature="API Integration",
            type="API",
            category="Contract",
            objective="Test API integration functionality"
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            generator = JavaScriptTestGenerator("jest", output_dir)
            
            generated_files = generator.generate([test_case])
            
            assert "integration.test.js" in generated_files
            assert "package.json" in generated_files
            assert "jest.config.js" in generated_files
            
            # Check that Jest syntax is present
            test_content = generated_files["integration.test.js"]
            assert "describe(" in test_content
            assert "test(" in test_content


class TestGeneratorUtils:
    """Test generator utility functions."""
    
    def test_sanitize_test_name(self):
        """Test test name sanitization."""
        from testwriter.generators.base_generator import BaseTestGenerator
        
        generator = BaseTestGenerator("test", Path("."))
        
        # Mock the abstract methods
        generator.generate = lambda x: {}
        generator.get_supported_frameworks = lambda: []
        generator.get_file_extension = lambda: ".test"
        
        assert generator.sanitize_test_name("E2E_[1]") == "e2e_1"
        assert generator.sanitize_test_name("User Login Test") == "user_login_test"
        assert generator.sanitize_test_name("API/Integration") == "api_integration"
        assert generator.sanitize_test_name("123_test") == "test_123_test"
    
    def test_categorize_tests(self):
        """Test test categorization."""
        from testwriter.generators.base_generator import BaseTestGenerator
        
        generator = BaseTestGenerator("test", Path("."))
        
        # Mock the abstract methods
        generator.generate = lambda x: {}
        generator.get_supported_frameworks = lambda: []
        generator.get_file_extension = lambda: ".test"
        
        test_cases = [
            TestCase(id="E2E_[1]", feature="", type="", category="", objective=""),
            TestCase(id="INT_[1]", feature="", type="", category="", objective=""),
            TestCase(id="TECH_[1]", feature="", type="", category="", objective=""),
            TestCase(id="MOCK_[1]", feature="", type="", category="", objective=""),
        ]
        
        categorized = generator.categorize_tests(test_cases)
        
        assert len(categorized['e2e']) == 1
        assert len(categorized['integration']) == 1
        assert len(categorized['technical']) == 1
        assert len(categorized['mocked']) == 1