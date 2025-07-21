"""Unit tests for testteller automator_agent CLI automation commands."""

import pytest

pytestmark = pytest.mark.automation
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import typer
from typer.testing import CliRunner

from testteller.automator_agent.cli import (
    automate_command,
    get_generator,
    validate_framework,
    interactive_select_tests,
    parse_selection,
    print_next_steps
)
from testteller.core.constants import SUPPORTED_LANGUAGES, SUPPORTED_FRAMEWORKS
from testteller.automator_agent.parser.markdown_parser import TestCase


class TestCLIFunctions:
    """Test CLI utility functions."""
    
    def test_validate_framework(self):
        """Test framework validation."""
        # Valid frameworks
        assert validate_framework("python", "pytest") == True
        assert validate_framework("python", "unittest") == True
        assert validate_framework("javascript", "jest") == True
        assert validate_framework("javascript", "mocha") == True
        assert validate_framework("java", "junit5") == True
        
        # Invalid frameworks
        assert validate_framework("python", "invalid") == False
        assert validate_framework("invalid", "pytest") == False
        assert validate_framework("javascript", "pytest") == False
    
    def test_get_generator(self):
        """Test getting appropriate generator."""
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Test Python generator
            python_gen = get_generator("python", "pytest", temp_dir)
            assert python_gen.__class__.__name__ == "PythonTestGenerator"
            assert python_gen.framework == "pytest"
            
            # Test JavaScript generator
            js_gen = get_generator("javascript", "jest", temp_dir)
            assert js_gen.__class__.__name__ == "JavaScriptTestGenerator"
            assert js_gen.framework == "jest"
            
            # Test Java generator
            java_gen = get_generator("java", "junit5", temp_dir)
            assert java_gen.__class__.__name__ == "JavaTestGenerator"
            assert java_gen.framework == "junit5"
            
            # Test unsupported language
            with pytest.raises(ValueError, match="Unsupported language"):
                get_generator("unsupported", "framework", temp_dir)
                
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_parse_selection(self):
        """Test parsing user selection strings."""
        # Single numbers
        assert parse_selection("1", 10) == [1]
        assert parse_selection("5", 10) == [5]
        
        # Multiple numbers
        assert set(parse_selection("1,3,5", 10)) == {1, 3, 5}
        assert set(parse_selection("1, 3, 5", 10)) == {1, 3, 5}  # With spaces
        
        # Ranges
        assert set(parse_selection("1-3", 10)) == {1, 2, 3}
        assert set(parse_selection("5-7", 10)) == {5, 6, 7}
        
        # Mixed
        assert set(parse_selection("1,3-5,8", 10)) == {1, 3, 4, 5, 8}
        
        # Out of range (should be filtered)
        assert parse_selection("15", 10) == []
        assert set(parse_selection("1,15,3", 10)) == {1, 3}
        
        # Invalid format (should be ignored)
        assert set(parse_selection("1,abc,3", 10)) == {1, 3}
        assert set(parse_selection("1,2-abc,3", 10)) == {1, 3}
    
    def test_interactive_select_tests(self):
        """Test interactive test selection."""
        test_cases = [
            TestCase(id="E2E_[1]", feature="", type="", category="", 
                    objective="Test user login functionality"),
            TestCase(id="INT_[1]", feature="", type="", category="", 
                    objective="Test API integration"),
            TestCase(id="TECH_[1]", feature="", type="", category="", 
                    objective="Test performance under load"),
        ]
        
        # Test selecting all
        with patch('typer.prompt', return_value='all'):
            selected = interactive_select_tests(test_cases)
            assert len(selected) == 3
            assert selected == test_cases
        
        # Test selecting none
        with patch('typer.prompt', return_value='none'):
            selected = interactive_select_tests(test_cases)
            assert len(selected) == 0
        
        # Test selecting specific tests
        with patch('typer.prompt', return_value='1,3'):
            selected = interactive_select_tests(test_cases)
            assert len(selected) == 2
            assert selected[0].id == "E2E_[1]"
            assert selected[1].id == "TECH_[1]"
    
    def test_print_next_steps(self, capsys):
        """Test printing next steps for different languages."""
        temp_dir = Path("/test/output")
        
        # Test Python
        print_next_steps("python", "pytest", temp_dir)
        captured = capsys.readouterr()
        assert "pip install -r requirements.txt" in captured.out
        assert "pytest" in captured.out
        
        # Test JavaScript
        print_next_steps("javascript", "jest", temp_dir)
        captured = capsys.readouterr()
        assert "npm install" in captured.out
        assert "npm test" in captured.out
        
        # Test Java
        print_next_steps("java", "junit5", temp_dir)
        captured = capsys.readouterr()
        assert "mvn clean install" in captured.out
        assert "mvn test" in captured.out


class TestAutomateCommand:
    """Test the main automate command."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.runner = CliRunner()
        
        # Create a sample test cases file
        self.test_cases_content = """
### Test Case E2E_[1]
**Feature:** User Login
**Type:** Authentication
**Category:** Happy Path

#### Objective
Verify that users can successfully log in with valid credentials.

#### Test Steps
1. **Action:** Navigate to login page
   - **Technical Details:** Send GET request to /login
2. **Validation:** Verify login form is displayed
   - **Technical Details:** Check for username and password fields

### Test Case INT_[1]
**Integration:** User Service -> Order Service
**Type:** API
**Category:** Contract

#### Objective
Verify that User Service correctly communicates with Order Service.
"""
        
        self.test_file = self.temp_dir / "test_cases.md"
        with open(self.test_file, 'w') as f:
            f.write(self.test_cases_content)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_automate_command_file_not_found(self):
        """Test automate command with non-existent file."""
        with pytest.raises(typer.Exit) as exc_info:
            automate_command(
                input_file="nonexistent.md",
                language="python",
                framework="pytest",
                output_dir="./output"
            )
        assert exc_info.value.exit_code == 1
    
    def test_automate_command_invalid_framework(self):
        """Test automate command with invalid framework."""
        with pytest.raises(typer.Exit) as exc_info:
            automate_command(
                input_file=str(self.test_file),
                language="python",
                framework="invalid",
                output_dir="./output"
            )
        assert exc_info.value.exit_code == 1
    
    @patch('typer.prompt')
    @patch('typer.confirm')
    def test_automate_command_interactive_mode(self, mock_confirm, mock_prompt):
        """Test automate command in interactive mode."""
        # Mock user selections
        mock_prompt.side_effect = [1, 1]  # Select python, then pytest
        
        output_dir = self.temp_dir / "output"
        
        # This should not raise an exception
        try:
            automate_command(
                input_file=str(self.test_file),
                language=None,  # Will be selected interactively
                framework=None,  # Will be selected interactively
                output_dir=str(output_dir),
                interactive=False
            )
        except typer.Exit as e:
            # If it exits with code 0, that's success
            if e.exit_code != 0:
                pytest.fail(f"Command failed with exit code {e.exit_code}")
    
    @patch('typer.prompt')
    def test_automate_command_interactive_selection(self, mock_prompt):
        """Test automate command with interactive test selection."""
        # Mock user selections for interactive test selection
        mock_prompt.return_value = "1,2"  # Select tests 1,2
        
        output_dir = self.temp_dir / "output"
        
        try:
            automate_command(
                input_file=str(self.test_file),
                language="python",  # Provide explicit language
                framework="pytest",  # Provide explicit framework
                output_dir=str(output_dir),
                interactive=True
            )
        except typer.Exit as e:
            if e.exit_code != 0:
                pytest.fail(f"Command failed with exit code {e.exit_code}")
    
    def test_automate_command_direct_generation(self):
        """Test automate command with direct parameters."""
        output_dir = self.temp_dir / "output"
        
        try:
            automate_command(
                input_file=str(self.test_file),
                language="python",
                framework="pytest",
                output_dir=str(output_dir),
                interactive=False
            )
            
            # Check that output directory was created and contains files
            assert output_dir.exists()
            
        except typer.Exit as e:
            if e.exit_code != 0:
                pytest.fail(f"Command failed with exit code {e.exit_code}")
    
    def test_automate_command_empty_test_file(self):
        """Test automate command with empty test file."""
        empty_file = self.temp_dir / "empty.md"
        with open(empty_file, 'w') as f:
            f.write("# Empty file\nNo test cases here.")
        
        with pytest.raises(typer.Exit) as exc_info:
            automate_command(
                input_file=str(empty_file),
                language="python",
                framework="pytest",
                output_dir="./output"
            )
        assert exc_info.value.exit_code == 1
    
    @patch('typer.prompt')
    def test_automate_command_user_abort(self, mock_prompt):
        """Test automate command when user aborts selection."""
        # Simulate user selecting 'none' in interactive mode
        mock_prompt.return_value = "none"
        
        with pytest.raises(typer.Exit) as exc_info:
            automate_command(
                input_file=str(self.test_file),
                language="python",
                framework="pytest",
                output_dir="./output",
                interactive=True
            )
        assert exc_info.value.exit_code == 1
    
    def test_automate_command_malformed_input_file(self):
        """Test automate command with malformed test cases file."""
        malformed_file = self.temp_dir / "malformed.md"
        with open(malformed_file, 'w') as f:
            f.write("""
# Not a test case
This file doesn't contain properly formatted test cases.

## Some section
Random content here.
""")
        
        with pytest.raises(typer.Exit) as exc_info:
            automate_command(
                input_file=str(malformed_file),
                language="python",
                framework="pytest",
                output_dir="./output"
            )
        assert exc_info.value.exit_code == 1


class TestSupportedLanguages:
    """Test supported languages configuration."""
    
    def test_supported_languages_structure(self):
        """Test that supported languages are properly configured."""
        assert isinstance(SUPPORTED_LANGUAGES, list)
        
        # Check expected languages
        assert "python" in SUPPORTED_LANGUAGES
        assert "javascript" in SUPPORTED_LANGUAGES
        assert "java" in SUPPORTED_LANGUAGES
        
        # Check frameworks structure
        assert isinstance(SUPPORTED_FRAMEWORKS, dict)
        
        # Check that each language has frameworks
        for language in SUPPORTED_LANGUAGES:
            assert language in SUPPORTED_FRAMEWORKS
            assert isinstance(SUPPORTED_FRAMEWORKS[language], list)
            assert len(SUPPORTED_FRAMEWORKS[language]) > 0
        
        # Check specific frameworks
        assert "pytest" in SUPPORTED_FRAMEWORKS["python"]
        assert "unittest" in SUPPORTED_FRAMEWORKS["python"]
        assert "jest" in SUPPORTED_FRAMEWORKS["javascript"]
        assert "junit5" in SUPPORTED_FRAMEWORKS["java"]