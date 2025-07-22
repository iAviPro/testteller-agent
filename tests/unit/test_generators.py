"""Unit tests for testteller automator_agent code generators."""

import pytest

pytestmark = pytest.mark.automation
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from testteller.automator_agent.parser.markdown_parser import TestCase, TestStep
from testteller.automator_agent.base_generator import BaseTestGenerator


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


# Note: The following test classes are disabled as the corresponding generator classes
# (PythonTestGenerator, JavaScriptTestGenerator, JavaTestGenerator) are not yet implemented
# TODO: Re-enable these tests once the generator implementations are added

# class TestPythonGenerator:
# class TestJavaScriptGenerator:  
# class TestJavaGenerator: