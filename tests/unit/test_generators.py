"""Unit tests for testteller automator_agent code generators."""

import pytest

pytestmark = pytest.mark.automation
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from testteller.automator_agent.parser.markdown_parser import TestCase, TestStep
from testteller.automator_agent.generators.base_generator import BaseTestGenerator
from testteller.automator_agent.generators.python_generator import PythonTestGenerator
from testteller.automator_agent.generators.javascript_generator import JavaScriptTestGenerator
from testteller.automator_agent.generators.java_generator import JavaTestGenerator


class MockGenerator(BaseTestGenerator):
    """Mock generator for testing base functionality."""
    
    def generate(self, test_cases):
        return {"test.txt": "mock content"}
    
    def get_supported_frameworks(self):
        return ["mock_framework"]
    
    def get_file_extension(self):
        return ".mock"


class TestBaseGenerator:
    """Test the base generator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.generator = MockGenerator("mock_framework", self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_categorize_tests(self):
        """Test test categorization by type."""
        test_cases = [
            TestCase(id="E2E_[1]", feature="", type="", category="", objective=""),
            TestCase(id="INT_[1]", feature="", type="", category="", objective=""),
            TestCase(id="INT_[2]", feature="", type="", category="", objective=""),
            TestCase(id="TECH_[1]", feature="", type="", category="", objective=""),
            TestCase(id="MOCK_[1]", feature="", type="", category="", objective=""),
        ]
        
        categorized = self.generator.categorize_tests(test_cases)
        
        assert len(categorized['e2e']) == 1
        assert len(categorized['integration']) == 2
        assert len(categorized['technical']) == 1
        assert len(categorized['mocked']) == 1
        assert categorized['e2e'][0].id == "E2E_[1]"
        assert categorized['integration'][0].id == "INT_[1]"
        assert categorized['integration'][1].id == "INT_[2]"
    
    def test_sanitize_test_name(self):
        """Test test name sanitization."""
        assert self.generator.sanitize_test_name("E2E_[1]") == "e2e__1"
        assert self.generator.sanitize_test_name("User Login Test") == "user_login_test"
        assert self.generator.sanitize_test_name("API/Integration") == "api_integration"
        assert self.generator.sanitize_test_name("Test-with-dashes") == "test_with_dashes"
        assert self.generator.sanitize_test_name("123_test") == "test_123_test"
        assert self.generator.sanitize_test_name("Special@#$%Characters") == "specialcharacters"
        assert self.generator.sanitize_test_name("") == ""
    
    def test_generate_test_description(self):
        """Test generating test descriptions."""
        test_case = TestCase(
            id="E2E_[1]",
            feature="User Login",
            type="Authentication",
            category="Happy Path",
            objective="Verify successful user login"
        )
        
        description = self.generator.generate_test_description(test_case)
        
        assert "Test ID: E2E_[1]" in description
        assert "Feature: User Login" in description
        assert "Type: Authentication" in description
        assert "Category: Happy Path" in description
        assert "Objective: Verify successful user login" in description
    
    def test_extract_test_data(self):
        """Test extracting test data from prerequisites."""
        test_case = TestCase(
            id="E2E_[1]",
            feature="",
            type="",
            category="",
            objective="",
            prerequisites={
                "test_data": "user_id: 123, email: test@example.com, active: true, balance: 100.50"
            }
        )
        
        test_data = self.generator.extract_test_data(test_case)
        
        assert test_data["user_id"] == 123
        assert test_data["email"] == "test@example.com"
        assert test_data["active"] == True
        assert test_data["balance"] == 100.50
    
    def test_extract_test_data_empty(self):
        """Test extracting test data when none provided."""
        test_case = TestCase(
            id="E2E_[1]",
            feature="",
            type="",
            category="",
            objective=""
        )
        
        test_data = self.generator.extract_test_data(test_case)
        assert test_data == {}
    
    def test_write_files(self):
        """Test writing generated files to disk."""
        generated_files = {
            "test1.py": "# Test file 1\nprint('test1')",
            "test2.py": "# Test file 2\nprint('test2')",
            "subdir/test3.py": "# Test file 3\nprint('test3')"
        }
        
        self.generator.write_files(generated_files)
        
        # Check files were written
        assert (self.temp_dir / "test1.py").exists()
        assert (self.temp_dir / "test2.py").exists()
        assert (self.temp_dir / "subdir" / "test3.py").exists()
        
        # Check content
        with open(self.temp_dir / "test1.py") as f:
            assert f.read() == "# Test file 1\nprint('test1')"


class TestPythonGenerator:
    """Test the Python test generator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_with_valid_framework(self):
        """Test initializing with valid framework."""
        generator = PythonTestGenerator("pytest", self.temp_dir)
        assert generator.framework == "pytest"
        assert generator.output_dir == self.temp_dir
    
    def test_init_with_invalid_framework(self):
        """Test initializing with invalid framework."""
        with pytest.raises(ValueError, match="Unsupported framework"):
            PythonTestGenerator("invalid", self.temp_dir)
    
    def test_get_supported_frameworks(self):
        """Test getting supported frameworks."""
        generator = PythonTestGenerator("pytest", self.temp_dir)
        frameworks = generator.get_supported_frameworks()
        assert "pytest" in frameworks
        assert "unittest" in frameworks
    
    def test_get_file_extension(self):
        """Test getting file extension."""
        generator = PythonTestGenerator("pytest", self.temp_dir)
        assert generator.get_file_extension() == '.py'
    
    def test_generate_pytest_tests(self):
        """Test generating pytest tests."""
        test_case = TestCase(
            id="E2E_[1]",
            feature="User Login",
            type="Authentication",
            category="Happy Path",
            objective="Test user login functionality",
            prerequisites={
                "test_data": "user_id: 123, email: test@example.com"
            },
            test_steps=[
                TestStep(
                    action="Navigate to login page",
                    technical_details="Send GET request to /login"
                ),
                TestStep(
                    action="Verify login form displayed",
                    validation="Verify login form displayed",
                    validation_details="Check for username and password fields"
                )
            ]
        )
        
        generator = PythonTestGenerator("pytest", self.temp_dir)
        generated_files = generator.generate([test_case])
        
        # Check expected files are generated
        assert "test_e2e.py" in generated_files
        assert "conftest.py" in generated_files
        assert "requirements.txt" in generated_files
        
        # Check test file content
        test_content = generated_files["test_e2e.py"]
        assert "import pytest" in test_content
        assert "def test_e2e__1" in test_content
        assert "user_id = 123" in test_content
        assert "email = \"test@example.com\"" in test_content
        
        # Check conftest content
        conftest_content = generated_files["conftest.py"]
        assert "import pytest" in conftest_content
        assert "@pytest.fixture" in conftest_content
        
        # Check requirements content
        requirements_content = generated_files["requirements.txt"]
        assert "pytest" in requirements_content
    
    def test_generate_unittest_tests(self):
        """Test generating unittest tests."""
        test_case = TestCase(
            id="INT_[1]",
            feature="API Integration",
            type="REST API",
            category="Contract",
            objective="Test API integration"
        )
        
        generator = PythonTestGenerator("unittest", self.temp_dir)
        generated_files = generator.generate([test_case])
        
        # Check expected files are generated
        assert "test_integration.py" in generated_files
        assert "requirements.txt" in generated_files
        
        # Check test file content
        test_content = generated_files["test_integration.py"]
        assert "import unittest" in test_content
        assert "class TestIntegration(unittest.TestCase)" in test_content
        assert "def test_int__1" in test_content
        assert "if __name__ == '__main__':" in test_content
        assert "unittest.main()" in test_content
    
    def test_extract_endpoint(self):
        """Test extracting endpoint from technical details."""
        generator = PythonTestGenerator("pytest", self.temp_dir)
        
        # Test with API endpoint
        endpoint = generator._extract_endpoint("Send PUT request to /api/v1/users/123")
        assert endpoint == "/api/v1/users/123"
        
        # Test without endpoint
        endpoint = generator._extract_endpoint("Some other technical detail")
        assert endpoint == "/api/endpoint"  # Default
    
    def test_extract_status_code(self):
        """Test extracting status code from text."""
        generator = PythonTestGenerator("pytest", self.temp_dir)
        
        assert generator._extract_status_code("200 OK") == "200"
        assert generator._extract_status_code("Expect 404 Not Found") == "404"
        assert generator._extract_status_code("No status code here") == "200"  # Default
    
    def test_step_code_generation(self):
        """Test generating code for different step types."""
        generator = PythonTestGenerator("pytest", self.temp_dir)
        test_case = TestCase(id="E2E_[1]", feature="", type="", category="", objective="")
        
        # Test API step
        api_step = TestStep(
            action="Send API request",
            technical_details="Send PUT request to /api/v1/profile"
        )
        code = generator._generate_step_code(api_step, test_case)
        assert "requests.put" in code
        assert "/api/v1/profile" in code
        
        # Test UI step
        ui_step = TestStep(action="Navigate to settings page")
        code = generator._generate_step_code(ui_step, test_case)
        assert "TODO: Implement UI interaction" in code
        
        # Test generic step
        generic_step = TestStep(action="Perform some action")
        code = generator._generate_step_code(generic_step, test_case)
        assert "TODO: Implement action" in code


class TestJavaScriptGenerator:
    """Test the JavaScript test generator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_with_valid_framework(self):
        """Test initializing with valid framework."""
        generator = JavaScriptTestGenerator("jest", self.temp_dir)
        assert generator.framework == "jest"
        assert generator.output_dir == self.temp_dir
    
    def test_init_with_invalid_framework(self):
        """Test initializing with invalid framework."""
        with pytest.raises(ValueError, match="Unsupported framework"):
            JavaScriptTestGenerator("invalid", self.temp_dir)
    
    def test_get_file_extension(self):
        """Test getting file extension for different frameworks."""
        jest_gen = JavaScriptTestGenerator("jest", self.temp_dir)
        assert jest_gen.get_file_extension() == ".test.js"
        
        playwright_gen = JavaScriptTestGenerator("playwright", self.temp_dir)
        assert playwright_gen.get_file_extension() == ".spec.js"
        
        cypress_gen = JavaScriptTestGenerator("cypress", self.temp_dir)
        assert cypress_gen.get_file_extension() == ".cy.js"
    
    def test_generate_jest_tests(self):
        """Test generating Jest tests."""
        test_case = TestCase(
            id="E2E_[1]",
            feature="User Registration",
            type="Form Submission",
            category="Happy Path",
            objective="Test user registration flow"
        )
        
        generator = JavaScriptTestGenerator("jest", self.temp_dir)
        generated_files = generator.generate([test_case])
        
        # Check expected files are generated
        assert "e2e.test.js" in generated_files
        assert "package.json" in generated_files
        assert "jest.config.js" in generated_files
        
        # Check test file content
        test_content = generated_files["e2e.test.js"]
        assert "describe('End-to-End Tests'" in test_content
        assert "test('" in test_content
        assert "beforeAll" in test_content
        assert "afterAll" in test_content
        
        # Check package.json content
        package_content = json.loads(generated_files["package.json"])
        assert "jest" in package_content["devDependencies"]
        assert "test" in package_content["scripts"]
    
    def test_generate_playwright_tests(self):
        """Test generating Playwright tests."""
        test_case = TestCase(
            id="E2E_[1]",
            feature="Navigation",
            type="UI",
            category="Happy Path",
            objective="Test page navigation"
        )
        
        generator = JavaScriptTestGenerator("playwright", self.temp_dir)
        generated_files = generator.generate([test_case])
        
        # Check expected files are generated
        assert "e2e.spec.js" in generated_files
        assert "package.json" in generated_files
        assert "playwright.config.js" in generated_files
        
        # Check test file content
        test_content = generated_files["e2e.spec.js"]
        assert "test('" in test_content
        assert "async ({ page })" in test_content
        assert "await page.goto" in test_content
    
    def test_generate_cypress_tests(self):
        """Test generating Cypress tests."""
        test_case = TestCase(
            id="E2E_[1]",
            feature="Shopping Cart",
            type="E2E Flow",
            category="Happy Path",
            objective="Test shopping cart functionality"
        )
        
        generator = JavaScriptTestGenerator("cypress", self.temp_dir)
        generated_files = generator.generate([test_case])
        
        # Check expected files are generated
        assert "e2e.cy.js" in generated_files
        assert "package.json" in generated_files
        assert "cypress.config.js" in generated_files
        
        # Check test file content
        test_content = generated_files["e2e.cy.js"]
        assert "describe('End-to-End Tests'" in test_content
        assert "it('" in test_content
        assert "cy.visit" in test_content
    
    def test_generate_test_name(self):
        """Test generating descriptive test names."""
        generator = JavaScriptTestGenerator("jest", self.temp_dir)
        
        # Test with objective
        test_case = TestCase(
            id="E2E_[1]",
            feature="",
            type="",
            category="",
            objective="Verify that users can successfully complete checkout process"
        )
        name = generator._generate_test_name(test_case)
        assert "Verify that users can successfully complete checkout process" == name
        
        # Test with long objective (should truncate)
        test_case.objective = "This is a very long objective that should be truncated because it exceeds the maximum length limit"
        name = generator._generate_test_name(test_case)
        assert len(name) <= 83  # 80 chars + "..."
        assert name.endswith("...")
        
        # Test without objective
        test_case.objective = ""
        name = generator._generate_test_name(test_case)
        assert name == "Test E2E_[1]"


class TestJavaGenerator:
    """Test the Java test generator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_with_valid_framework(self):
        """Test initializing with valid framework."""
        generator = JavaTestGenerator("junit5", self.temp_dir)
        assert generator.framework == "junit5"
        assert generator.output_dir == self.temp_dir
    
    def test_init_with_invalid_framework(self):
        """Test initializing with invalid framework."""
        with pytest.raises(ValueError, match="Unsupported framework"):
            JavaTestGenerator("invalid", self.temp_dir)
    
    def test_get_file_extension(self):
        """Test getting file extension."""
        generator = JavaTestGenerator("junit5", self.temp_dir)
        assert generator.get_file_extension() == '.java'
    
    def test_generate_junit5_tests(self):
        """Test generating JUnit 5 tests."""
        test_case = TestCase(
            id="E2E_[1]",
            feature="User Management",
            type="CRUD Operations",
            category="Happy Path",
            objective="Test user CRUD operations"
        )
        
        generator = JavaTestGenerator("junit5", self.temp_dir)
        generated_files = generator.generate([test_case])
        
        # Check expected files are generated
        assert "EndToEndTest.java" in generated_files
        assert "pom.xml" in generated_files
        assert "TestBase.java" in generated_files
        
        # Check test file content
        test_content = generated_files["EndToEndTest.java"]
        assert "package com.testteller.generated;" in test_content
        assert "import org.junit.jupiter.api.*" in test_content
        assert "public class EndToEndTest extends TestBase" in test_content
        assert "@BeforeEach" in test_content
        assert "@AfterEach" in test_content
        assert "@Test" in test_content
        assert "public void teste2e1()" in test_content
        
        # Check pom.xml content
        pom_content = generated_files["pom.xml"]
        assert "junit-jupiter" in pom_content
        assert "<version>5.9.0</version>" in pom_content
    
    def test_generate_junit4_tests(self):
        """Test generating JUnit 4 tests."""
        test_case = TestCase(
            id="INT_[1]",
            feature="API",
            type="Integration",
            category="Contract",
            objective="Test API integration"
        )
        
        generator = JavaTestGenerator("junit4", self.temp_dir)
        generated_files = generator.generate([test_case])
        
        # Check test file content
        test_content = generated_files["IntegrationTest.java"]
        assert "import org.junit.*" in test_content
        assert "@Before" in test_content
        assert "@After" in test_content
        assert "@Test" in test_content
        
        # Check pom.xml content
        pom_content = generated_files["pom.xml"]
        assert "junit" in pom_content
        assert "<version>4.13.2</version>" in pom_content
    
    def test_generate_testng_tests(self):
        """Test generating TestNG tests."""
        test_case = TestCase(
            id="TECH_[1]",
            feature="Performance",
            type="Load Test",
            category="Performance",
            objective="Test system performance under load"
        )
        
        generator = JavaTestGenerator("testng", self.temp_dir)
        generated_files = generator.generate([test_case])
        
        # Check test file content
        test_content = generated_files["TechnicalTest.java"]
        assert "import org.testng.annotations.*" in test_content
        assert "@BeforeMethod" in test_content
        assert "@AfterMethod" in test_content
        assert '@Test(description = "Test system performance under load")' in test_content
        
        # Check pom.xml content
        pom_content = generated_files["pom.xml"]
        assert "testng" in pom_content
    
    def test_camel_case_conversion(self):
        """Test converting text to camelCase."""
        generator = JavaTestGenerator("junit5", self.temp_dir)
        
        assert generator._to_camel_case("E2E_[1]") == "e2e1"
        assert generator._to_camel_case("user_login_test") == "userLoginTest"
        assert generator._to_camel_case("simple") == "simple"
        assert generator._to_camel_case("UPPER_CASE") == "upperCase"
    
    def test_infer_java_type(self):
        """Test inferring Java types from Python values."""
        generator = JavaTestGenerator("junit5", self.temp_dir)
        
        assert generator._infer_java_type(True) == "boolean"
        assert generator._infer_java_type(123) == "int"
        assert generator._infer_java_type(123.45) == "double"
        assert generator._infer_java_type("test") == "String"
        assert generator._infer_java_type(None) == "String"  # Default
    
    def test_get_category_class_name(self):
        """Test getting Java class names for categories."""
        generator = JavaTestGenerator("junit5", self.temp_dir)
        
        assert generator._get_category_class_name("e2e") == "EndToEnd"
        assert generator._get_category_class_name("integration") == "Integration"
        assert generator._get_category_class_name("technical") == "Technical"
        assert generator._get_category_class_name("mocked") == "Mocked"
        assert generator._get_category_class_name("unknown") == "Test"