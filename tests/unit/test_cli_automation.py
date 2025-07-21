"""Unit tests for testteller automator_agent RAG-enhanced CLI automation commands."""

import pytest

pytestmark = pytest.mark.automation
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import typer
from typer.testing import CliRunner

from testteller.automator_agent.cli import (
    automate_command,
    validate_framework,
    interactive_select_tests,
    parse_selection,
    print_next_steps,
    get_collection_name,
    get_language,
    get_framework,
    initialize_vector_store,
    assess_generated_quality
)
from testteller.core.constants import SUPPORTED_LANGUAGES, SUPPORTED_FRAMEWORKS
from testteller.automator_agent.parser.markdown_parser import TestCase
from testteller.automator_agent.rag_enhanced_generator import RAGEnhancedTestGenerator
from testteller.automator_agent.application_context import ApplicationContext


class TestRAGEnhancedCLIFunctions:
    """Test RAG-enhanced CLI utility functions."""
    
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
        
    def test_get_collection_name(self):
        """Test collection name resolution."""
        # Test with provided name
        assert get_collection_name("custom_collection") == "custom_collection"
        
        # Test with default fallback
        assert get_collection_name(None) == "test_collection"
        
    def test_get_language(self):
        """Test language resolution."""
        # Test with provided language
        assert get_language("javascript") == "javascript"
        
        # Test with environment variable
        with patch.dict('os.environ', {'AUTOMATION_LANGUAGE': 'typescript'}):
            assert get_language(None) == 'typescript'
        
        # Test default fallback
        assert get_language(None) == "python"
        
    def test_get_framework(self):
        """Test framework resolution."""
        # Test with provided framework
        assert get_framework("jest", "javascript") == "jest"
        
        # Test with environment variable
        with patch.dict('os.environ', {'AUTOMATION_FRAMEWORK': 'cypress'}):
            assert get_framework(None, "javascript") == 'cypress'
        
        # Test default fallback for language
        assert get_framework(None, "python") == "pytest"
        
    @patch('testteller.automator_agent.cli.LLMManager')
    @patch('testteller.automator_agent.cli.ChromaDBManager')
    def test_initialize_vector_store(self, mock_chroma, mock_llm):
        """Test vector store initialization."""
        mock_vector_store = Mock()
        mock_chroma.return_value = mock_vector_store
        mock_vector_store.list_collections.return_value = ['collection1', 'collection2']
        
        result = initialize_vector_store("test_collection")
        
        assert result == mock_vector_store
        mock_chroma.assert_called_once()
        
    def test_assess_generated_quality(self, capsys):
        """Test quality assessment of generated files."""
        generated_files = {
            "test_login.py": """import pytest
import requests

def test_login_success():
    response = requests.post('/api/login', json={'email': 'test@test.com', 'password': 'pass'})
    assert response.status_code == 200
    
def test_login_invalid():
    response = requests.post('/api/login', json={'email': 'invalid', 'password': 'invalid'})
    assert response.status_code == 401
""",
            "test_signup.py": """import pytest

def test_signup():
    # TODO: Implement signup test
    pass
"""
        }
        
        assess_generated_quality(generated_files)
        captured = capsys.readouterr()
        
        assert "Quality Assessment:" in captured.out
        assert "Total Lines of Code:" in captured.out
        assert "Test Functions:" in captured.out
        assert "TODO Items:" in captured.out
        assert "Quality Score:" in captured.out
    
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


class TestRAGEnhancedAutomateCommand:
    """Test the main RAG-enhanced automate command."""
    
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
    
    @patch('testteller.automator_agent.cli.initialize_vector_store')
    @patch('testteller.automator_agent.cli.LLMManager')
    @patch('testteller.automator_agent.cli.UnifiedDocumentParser')
    @patch('testteller.automator_agent.cli.RAGEnhancedTestGenerator')
    def test_automate_command_with_vector_store_context(self, mock_rag_gen, mock_parser, mock_llm, mock_vector_store):
        """Test automate command uses vector store for context."""
        output_dir = self.temp_dir / "output"
        
        # Mock vector store with application context
        mock_vs = Mock()
        mock_vector_store.return_value = mock_vs
        mock_vs.list_collections.return_value = ['test_collection']
        
        # Mock LLM manager
        mock_llm.return_value = Mock(provider="claude")
        
        # Mock parser
        mock_parsed_doc = Mock()
        mock_parsed_doc.test_cases = [
            TestCase(id="API_1", feature="UserAPI", type="API", category="Integration",
                    objective="Test user API endpoints", test_steps=[])
        ]
        mock_parsed_doc.metadata = Mock(title="API Tests", word_count=150)
        
        mock_parser_instance = Mock()
        mock_parser.return_value = mock_parser_instance
        mock_parser_instance.parse_for_automation = AsyncMock(return_value=mock_parsed_doc)
        
        # Mock RAG generator with context-aware generation
        mock_generator = Mock()
        mock_rag_gen.return_value = mock_generator
        mock_generator.generate.return_value = {
            "test_user_api.py": """import pytest
import requests

def test_get_user(base_url, auth_token):
    response = requests.get(f"{base_url}/api/users/1", 
                          headers={"Authorization": f"Bearer {auth_token}"})
    assert response.status_code == 200
    assert "id" in response.json()
"""
        }
        mock_generator.write_files.return_value = None
        
        try:
            automate_command(
                input_file=str(self.test_file),
                collection_name="test_collection",
                language="python",
                framework="pytest",
                output_dir=str(output_dir),
                interactive=False,
                num_context_docs=10,
                verbose=True
            )
            
            # Verify RAG generator was created with vector store
            mock_rag_gen.assert_called_once()
            call_args = mock_rag_gen.call_args
            assert call_args[1]['vector_store'] == mock_vs
            assert call_args[1]['num_context_docs'] == 10
            
        except typer.Exit as e:
            if e.exit_code != 0:
                pytest.fail(f"Command failed with exit code {e.exit_code}")
    
    @patch('testteller.automator_agent.cli.initialize_vector_store')
    @patch('testteller.automator_agent.cli.LLMManager')
    @patch('testteller.automator_agent.cli.UnifiedDocumentParser')
    @patch('testteller.automator_agent.cli.RAGEnhancedTestGenerator')
    @patch('typer.prompt')
    def test_automate_command_interactive_selection(self, mock_prompt, mock_rag_gen, mock_parser, mock_llm, mock_vector_store):
        """Test automate command with interactive test selection."""
        # Mock user selections for interactive test selection
        mock_prompt.return_value = "1,2"  # Select tests 1,2
        
        output_dir = self.temp_dir / "output"
        
        # Mock components
        mock_vector_store.return_value = Mock()
        mock_llm.return_value = Mock(provider="openai")
        
        # Mock parser with multiple test cases
        mock_parsed_doc = Mock()
        mock_parsed_doc.test_cases = [
            TestCase(id="E2E_1", feature="Login", type="E2E", category="Auth",
                    objective="Test login flow", test_steps=[]),
            TestCase(id="E2E_2", feature="Dashboard", type="E2E", category="UI",
                    objective="Test dashboard display", test_steps=[]),
            TestCase(id="INT_1", feature="API", type="Integration", category="API",
                    objective="Test API integration", test_steps=[])
        ]
        mock_parsed_doc.metadata = Mock(title="E2E Tests", word_count=200)
        
        mock_parser_instance = Mock()
        mock_parser.return_value = mock_parser_instance
        mock_parser_instance.parse_for_automation = AsyncMock(return_value=mock_parsed_doc)
        
        # Mock generator
        mock_generator = Mock()
        mock_rag_gen.return_value = mock_generator
        mock_generator.generate.return_value = {
            "test_e2e.py": "# E2E test code with real selectors"
        }
        mock_generator.write_files.return_value = None
        
        try:
            automate_command(
                input_file=str(self.test_file),
                collection_name="test_collection",
                language="python",
                framework="playwright",
                output_dir=str(output_dir),
                interactive=True,
                num_context_docs=5,
                verbose=False
            )
            
            # Verify interactive selection was handled
            mock_prompt.assert_called_once()
            
        except typer.Exit as e:
            if e.exit_code != 0:
                pytest.fail(f"Command failed with exit code {e.exit_code}")
    
    @patch('testteller.automator_agent.cli.initialize_vector_store')
    @patch('testteller.automator_agent.cli.LLMManager')
    @patch('testteller.automator_agent.cli.UnifiedDocumentParser')
    @patch('testteller.automator_agent.cli.RAGEnhancedTestGenerator')
    def test_automate_command_direct_generation(self, mock_rag_gen, mock_parser, mock_llm, mock_vector_store):
        """Test automate command with RAG-enhanced generation."""
        output_dir = self.temp_dir / "output"
        
        # Mock vector store
        mock_vector_store.return_value = Mock()
        
        # Mock LLM manager
        mock_llm.return_value = Mock(provider="gemini")
        
        # Mock parser to return test cases
        mock_parsed_doc = Mock()
        mock_parsed_doc.test_cases = [
            TestCase(id="TEST_1", feature="Login", type="E2E", category="Authentication",
                    objective="Test user login", test_steps=[])
        ]
        mock_parsed_doc.metadata = Mock(title="Test Cases", word_count=100)
        
        mock_parser_instance = Mock()
        mock_parser.return_value = mock_parser_instance
        mock_parser_instance.parse_for_automation = AsyncMock(return_value=mock_parsed_doc)
        
        # Mock RAG generator
        mock_generator = Mock()
        mock_rag_gen.return_value = mock_generator
        mock_generator.generate.return_value = {"test_login.py": "# Generated test code"}
        mock_generator.write_files.return_value = None
        
        try:
            automate_command(
                input_file=str(self.test_file),
                collection_name="test_collection",
                language="python",
                framework="pytest",
                output_dir=str(output_dir),
                interactive=False,
                num_context_docs=5,
                verbose=False
            )
            
            # Verify RAG components were called
            mock_vector_store.assert_called_once()
            mock_llm.assert_called_once()
            mock_rag_gen.assert_called_once()
            mock_generator.generate.assert_called_once()
            mock_generator.write_files.assert_called_once()
            
        except typer.Exit as e:
            if e.exit_code != 0:
                pytest.fail(f"Command failed with exit code {e.exit_code}")
    
    @patch('testteller.automator_agent.cli.initialize_vector_store')
    @patch('testteller.automator_agent.cli.LLMManager')
    @patch('testteller.automator_agent.cli.UnifiedDocumentParser')
    def test_automate_command_empty_test_file(self, mock_parser, mock_llm, mock_vector_store):
        """Test automate command with empty test file."""
        empty_file = self.temp_dir / "empty.md"
        with open(empty_file, 'w') as f:
            f.write("# Empty file\nNo test cases here.")
        
        # Mock components
        mock_vector_store.return_value = Mock()
        mock_llm.return_value = Mock(provider="openai")
        
        # Mock parser to return empty test cases
        mock_parsed_doc = Mock()
        mock_parsed_doc.test_cases = []
        mock_parsed_doc.metadata = Mock(title="Empty", word_count=10)
        
        mock_parser_instance = Mock()
        mock_parser.return_value = mock_parser_instance
        mock_parser_instance.parse_for_automation = AsyncMock(return_value=mock_parsed_doc)
        
        with pytest.raises(typer.Exit) as exc_info:
            automate_command(
                input_file=str(empty_file),
                collection_name="test_collection",
                language="python",
                framework="pytest",
                output_dir="./output",
                interactive=False,
                num_context_docs=5,
                verbose=False
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
    
    @patch('testteller.automator_agent.cli.initialize_vector_store')
    @patch('testteller.automator_agent.cli.LLMManager')
    @patch('testteller.automator_agent.cli.UnifiedDocumentParser')
    def test_automate_command_malformed_input_file(self, mock_parser, mock_llm, mock_vector_store):
        """Test automate command with malformed test cases file."""
        malformed_file = self.temp_dir / "malformed.md"
        with open(malformed_file, 'w') as f:
            f.write("""
# Not a test case
This file doesn't contain properly formatted test cases.

## Some section
Random content here.
""")
        
        # Mock components
        mock_vector_store.return_value = Mock()
        mock_llm.return_value = Mock(provider="gemini")
        
        # Mock parser to return empty test cases (simulating malformed file)
        mock_parsed_doc = Mock()
        mock_parsed_doc.test_cases = []
        mock_parsed_doc.metadata = Mock(title="Empty", word_count=50)
        
        mock_parser_instance = Mock()
        mock_parser.return_value = mock_parser_instance
        mock_parser_instance.parse_for_automation = AsyncMock(return_value=mock_parsed_doc)
        
        with pytest.raises(typer.Exit) as exc_info:
            automate_command(
                input_file=str(malformed_file),
                collection_name="test_collection",
                language="python",
                framework="pytest",
                output_dir="./output",
                interactive=False,
                num_context_docs=5,
                verbose=False
            )
        assert exc_info.value.exit_code == 1


class TestRAGEnhancedSupport:
    """Test RAG-enhanced automation support."""
    
    def test_supported_languages_structure(self):
        """Test that supported languages are properly configured for RAG enhancement."""
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
        
        # Check RAG-enhanced frameworks
        assert "pytest" in SUPPORTED_FRAMEWORKS["python"]
        assert "playwright" in SUPPORTED_FRAMEWORKS["python"]
        assert "unittest" in SUPPORTED_FRAMEWORKS["python"]
        assert "jest" in SUPPORTED_FRAMEWORKS["javascript"]
        assert "playwright" in SUPPORTED_FRAMEWORKS["javascript"]
        assert "junit5" in SUPPORTED_FRAMEWORKS["java"]
    
    @patch('testteller.automator_agent.cli.RAGEnhancedTestGenerator')
    def test_rag_generator_initialization(self, mock_rag_gen):
        """Test RAG generator is properly initialized with required components."""
        from testteller.automator_agent.cli import automate_command
        
        # Test that RAGEnhancedTestGenerator can be imported and mocked
        assert mock_rag_gen is not None
        
        # Test generator supports required languages and frameworks
        mock_generator = Mock()
        mock_generator.get_supported_frameworks.return_value = ['pytest', 'playwright', 'unittest']
        mock_generator.get_file_extension.return_value = '.py'
        mock_rag_gen.return_value = mock_generator
        
        # Verify mock setup
        generator_instance = mock_rag_gen(
            framework='pytest',
            output_dir=Path('/tmp'),
            vector_store=Mock(),
            language='python',
            llm_manager=Mock(),
            num_context_docs=5
        )
        
        assert generator_instance.get_supported_frameworks() == ['pytest', 'playwright', 'unittest']
        assert generator_instance.get_file_extension() == '.py'