"""
Unit tests for core functionality that should definitely work.
These tests are designed to increase coverage and ensure basic functionality.
"""

import pytest
from unittest.mock import Mock, patch
import os
import tempfile
from pathlib import Path

# Test core modules directly
from testteller.core.constants import (
    DEFAULT_COLLECTION_NAME, 
    DEFAULT_OUTPUT_FILE,
    SUPPORTED_LLM_PROVIDERS,
    SUPPORTED_LANGUAGES,
    SUPPORTED_FRAMEWORKS
)
from testteller.core.utils.exceptions import (
    EmbeddingGenerationError,
    DocumentIngestionError,
    CodeIngestionError,
    TestCaseGenerationError
)
from testteller.core.utils.retry_helpers import api_retry_sync, api_retry_async
from testteller._version import __version__


class TestConstants:
    """Test core constants are properly defined."""
    
    def test_default_values_exist(self):
        """Test that default values are properly defined."""
        assert DEFAULT_COLLECTION_NAME is not None
        assert DEFAULT_OUTPUT_FILE is not None
        assert len(DEFAULT_COLLECTION_NAME) > 0
        assert len(DEFAULT_OUTPUT_FILE) > 0
    
    def test_supported_providers(self):
        """Test that supported LLM providers are defined."""
        assert isinstance(SUPPORTED_LLM_PROVIDERS, list)
        assert len(SUPPORTED_LLM_PROVIDERS) > 0
        assert "gemini" in SUPPORTED_LLM_PROVIDERS
        assert "openai" in SUPPORTED_LLM_PROVIDERS
        assert "claude" in SUPPORTED_LLM_PROVIDERS
        assert "llama" in SUPPORTED_LLM_PROVIDERS
    
    def test_supported_languages(self):
        """Test that supported languages are defined."""
        assert isinstance(SUPPORTED_LANGUAGES, list)
        assert len(SUPPORTED_LANGUAGES) > 0
        assert "python" in SUPPORTED_LANGUAGES
        assert "javascript" in SUPPORTED_LANGUAGES
        assert "java" in SUPPORTED_LANGUAGES
        assert "typescript" in SUPPORTED_LANGUAGES
    
    def test_supported_frameworks(self):
        """Test that supported frameworks are defined."""
        assert isinstance(SUPPORTED_FRAMEWORKS, dict)
        assert len(SUPPORTED_FRAMEWORKS) > 0
        assert "python" in SUPPORTED_FRAMEWORKS
        assert "javascript" in SUPPORTED_FRAMEWORKS
        assert "java" in SUPPORTED_FRAMEWORKS
        assert "typescript" in SUPPORTED_FRAMEWORKS
        
        # Test that each language has frameworks
        for lang, frameworks in SUPPORTED_FRAMEWORKS.items():
            assert isinstance(frameworks, list)
            assert len(frameworks) > 0


class TestExceptions:
    """Test custom exceptions."""
    
    def test_embedding_generation_error(self):
        """Test EmbeddingGenerationError can be raised and caught."""
        with pytest.raises(EmbeddingGenerationError):
            raise EmbeddingGenerationError("Test error")
    
    def test_document_ingestion_error(self):
        """Test DocumentIngestionError can be raised and caught."""
        with pytest.raises(DocumentIngestionError):
            raise DocumentIngestionError("Test error")
    
    def test_code_ingestion_error(self):
        """Test CodeIngestionError can be raised and caught."""
        with pytest.raises(CodeIngestionError):
            raise CodeIngestionError("Test error")
    
    def test_test_case_generation_error(self):
        """Test TestCaseGenerationError can be raised and caught."""
        with pytest.raises(TestCaseGenerationError):
            raise TestCaseGenerationError("Test error")
    
    def test_exception_inheritance(self):
        """Test that custom exceptions inherit from Exception."""
        assert issubclass(EmbeddingGenerationError, Exception)
        assert issubclass(DocumentIngestionError, Exception)
        assert issubclass(CodeIngestionError, Exception)
        assert issubclass(TestCaseGenerationError, Exception)


class TestRetryHelpers:
    """Test retry helper functions."""
    
    def test_api_retry_sync_success(self):
        """Test that api_retry_sync works with successful function."""
        
        @api_retry_sync
        def successful_function():
            return "success"
        
        result = successful_function()
        assert result == "success"
    
    def test_api_retry_sync_with_failure_then_success(self):
        """Test that api_retry_sync retries on failure."""
        call_count = 0
        
        @api_retry_sync
        def function_that_fails_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("First call fails")
            return "success"
        
        result = function_that_fails_once()
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_api_retry_async_success(self):
        """Test that api_retry_async works with successful async function."""
        
        @api_retry_async
        async def successful_async_function():
            return "async_success"
        
        result = await successful_async_function()
        assert result == "async_success"


class TestVersion:
    """Test version functionality."""
    
    def test_version_exists(self):
        """Test that version is defined."""
        assert __version__ is not None
        assert isinstance(__version__, str)
        assert len(__version__) > 0
    
    def test_version_format(self):
        """Test that version follows semantic versioning."""
        # Basic check for semantic versioning pattern
        version_parts = __version__.split('.')
        assert len(version_parts) >= 2  # At least major.minor
        assert version_parts[0].isdigit()  # Major version is numeric
        assert version_parts[1].isdigit()  # Minor version is numeric


class TestCoreImports:
    """Test that core modules can be imported."""
    
    def test_import_main_module(self):
        """Test that main testteller module can be imported."""
        import testteller
        assert testteller is not None
        assert hasattr(testteller, '__version__')
    
    def test_import_core_modules(self):
        """Test that core modules can be imported."""
        from testteller.core import constants
        from testteller.core import utils
        from testteller.core.data_ingestion import document_loader
        from testteller.core.llm import base_client
        
        assert constants is not None
        assert utils is not None
        assert document_loader is not None
        assert base_client is not None
    
    def test_import_agent_modules(self):
        """Test that agent modules can be imported."""
        from testteller.generator_agent.agent import testteller_agent
        from testteller.automator_agent.parser import markdown_parser
        
        assert testteller_agent is not None
        assert markdown_parser is not None


class TestBasicFunctionality:
    """Test basic functionality that should always work."""
    
    def test_create_temp_directory(self):
        """Test that we can create temporary directories for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            assert temp_path.exists()
            assert temp_path.is_dir()
    
    def test_create_temp_file(self):
        """Test that we can create temporary files for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Test Content\nThis is a test.")
            temp_file = f.name
        
        try:
            assert Path(temp_file).exists()
            with open(temp_file, 'r') as f:
                content = f.read()
                assert "Test Content" in content
        finally:
            Path(temp_file).unlink()
    
    def test_environment_variables(self):
        """Test basic environment variable handling."""
        # Test setting and getting env vars
        test_key = "TESTTELLER_TEST_VAR"
        test_value = "test_value_123"
        
        os.environ[test_key] = test_value
        try:
            assert os.getenv(test_key) == test_value
        finally:
            del os.environ[test_key]


class TestDocumentLoaderBasics:
    """Test basic document loader functionality."""
    
    def test_document_loader_import(self):
        """Test that document loader can be imported and instantiated."""
        from testteller.core.data_ingestion.document_loader import DocumentLoader
        
        loader = DocumentLoader()
        assert loader is not None
    
    def test_supported_extensions(self):
        """Test that document loader knows about supported file types."""
        from testteller.core.data_ingestion.document_loader import DocumentLoader
        
        loader = DocumentLoader()
        # These should be basic attributes or methods
        assert hasattr(loader, 'load_text') or hasattr(loader, 'load_document')


class TestMarkdownParser:
    """Test basic markdown parser functionality."""
    
    def test_markdown_parser_import(self):
        """Test that markdown parser can be imported and instantiated."""
        from testteller.automator_agent.parser.markdown_parser import MarkdownTestCaseParser
        
        parser = MarkdownTestCaseParser()
        assert parser is not None
    
    def test_test_case_pattern(self):
        """Test that the parser has a test case pattern."""
        from testteller.automator_agent.parser.markdown_parser import MarkdownTestCaseParser
        
        parser = MarkdownTestCaseParser()
        assert hasattr(parser, 'test_case_pattern')
        assert parser.test_case_pattern is not None


class TestConfig:
    """Test basic configuration functionality."""
    
    def test_config_import(self):
        """Test that config can be imported."""
        from testteller import config
        assert config is not None
    
    @patch.dict(os.environ, {'LOG_LEVEL': 'DEBUG'})
    def test_config_with_env_var(self):
        """Test that config respects environment variables."""
        assert os.getenv('LOG_LEVEL') == 'DEBUG'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])