"""
Unit tests for TestTellerAgent class.
"""
import pytest
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from testteller.agent.testteller_agent import TestTellerAgent


class TestTestTellerAgent:
    """Test cases for TestTellerAgent class."""

    @pytest.mark.unit
    def test_init_with_default_params(self, mock_env_vars):
        """Test TestTellerAgent initialization with default parameters."""
        with patch.dict(os.environ, mock_env_vars):
            with patch('testteller.agent.testteller_agent.LLMManager') as mock_llm:
                with patch('testteller.agent.testteller_agent.ChromaDBManager') as mock_chroma:
                    mock_llm.return_value = Mock()
                    mock_chroma.return_value = Mock()

                    agent = TestTellerAgent()

                    assert agent.collection_name is not None
                    assert agent.llm_manager is not None
                    assert agent.vector_store is not None
                    assert agent.document_loader is not None
                    assert agent.code_loader is not None

    @pytest.mark.unit
    def test_init_with_custom_params(self, mock_env_vars, mock_llm_manager):
        """Test TestTellerAgent initialization with custom parameters."""
        with patch.dict(os.environ, mock_env_vars):
            with patch('testteller.agent.testteller_agent.ChromaDBManager') as mock_chroma:
                mock_chroma.return_value = Mock()

                agent = TestTellerAgent(
                    collection_name="custom_collection",
                    llm_manager=mock_llm_manager
                )

                assert agent.collection_name == "custom_collection"
                assert agent.llm_manager == mock_llm_manager

    @pytest.mark.unit
    def test_get_collection_name_from_settings(self, mock_env_vars):
        """Test getting collection name from settings."""
        with patch.dict(os.environ, mock_env_vars):
            with patch('testteller.agent.testteller_agent.settings') as mock_settings:
                mock_chromadb_settings = Mock()
                mock_chromadb_settings.default_collection_name = "settings_collection"
                mock_settings.chromadb = mock_chromadb_settings

                with patch('testteller.agent.testteller_agent.LLMManager'):
                    with patch('testteller.agent.testteller_agent.ChromaDBManager'):
                        agent = TestTellerAgent()
                        assert agent.collection_name == "settings_collection"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_ingest_documents_from_path_single_file(self, mock_testteller_agent, temp_dir):
        """Test ingesting documents from a single file."""
        # Create a test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("Test content")

        # Mock document loader
        mock_testteller_agent.document_loader.load_document = AsyncMock(
            return_value="Test content")

        await mock_testteller_agent.ingest_documents_from_path(str(test_file))

        # Verify document loader was called
        mock_testteller_agent.document_loader.load_document.assert_called_once_with(
            str(test_file))

        # Verify vector store was called
        mock_testteller_agent.vector_store.add_documents.assert_called_once()
        args = mock_testteller_agent.vector_store.add_documents.call_args
        assert args[0][0] == ["Test content"]  # documents
        assert args[0][1][0]["source"] == str(test_file)  # metadata
        assert len(args[0][2]) == 1  # ids

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_ingest_documents_from_path_directory(self, mock_testteller_agent, temp_dir):
        """Test ingesting documents from a directory."""
        # Create test directory with files
        test_dir = temp_dir / "docs"
        test_dir.mkdir()
        (test_dir / "doc1.txt").write_text("Content 1")
        (test_dir / "doc2.txt").write_text("Content 2")

        # Mock document loader
        mock_testteller_agent.document_loader.load_from_directory = AsyncMock(
            return_value=[("Content 1", "doc1.txt"), ("Content 2", "doc2.txt")]
        )

        await mock_testteller_agent.ingest_documents_from_path(str(test_dir))

        # Verify document loader was called
        mock_testteller_agent.document_loader.load_from_directory.assert_called_once_with(
            str(test_dir))

        # Verify vector store was called
        mock_testteller_agent.vector_store.add_documents.assert_called_once()
        args = mock_testteller_agent.vector_store.add_documents.call_args
        assert args[0][0] == ["Content 1", "Content 2"]  # documents
        assert len(args[0][1]) == 2  # metadata
        assert len(args[0][2]) == 2  # ids

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_ingest_documents_from_path_no_content(self, mock_testteller_agent, temp_dir):
        """Test ingesting documents when no content is found."""
        test_file = temp_dir / "empty.txt"
        test_file.write_text("")

        # Mock document loader to return None
        mock_testteller_agent.document_loader.load_document = AsyncMock(
            return_value=None)

        await mock_testteller_agent.ingest_documents_from_path(str(test_file))

        # Verify vector store was not called
        mock_testteller_agent.vector_store.add_documents.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_ingest_documents_from_path_error(self, mock_testteller_agent, temp_dir):
        """Test error handling during document ingestion."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Test content")

        # Mock document loader to raise exception
        mock_testteller_agent.document_loader.load_document = AsyncMock(
            side_effect=Exception("Load error")
        )

        with pytest.raises(Exception, match="Load error"):
            await mock_testteller_agent.ingest_documents_from_path(str(test_file))

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_ingest_code_from_source_remote(self, mock_testteller_agent):
        """Test ingesting code from remote repository."""
        repo_url = "https://github.com/test/repo.git"

        # Mock code loader
        mock_testteller_agent.code_loader.load_code_from_repo = AsyncMock(
            return_value=[("def test():\n    pass", "test.py")]
        )
        mock_testteller_agent.code_loader.cleanup_repo = AsyncMock()

        await mock_testteller_agent.ingest_code_from_source(repo_url)

        # Verify code loader was called
        mock_testteller_agent.code_loader.load_code_from_repo.assert_called_once_with(
            repo_url)
        mock_testteller_agent.code_loader.cleanup_repo.assert_called_once_with(
            repo_url)

        # Verify vector store was called
        mock_testteller_agent.vector_store.add_documents.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_ingest_code_from_source_local(self, mock_testteller_agent, temp_dir):
        """Test ingesting code from local folder."""
        local_path = str(temp_dir)

        # Mock code loader
        mock_testteller_agent.code_loader.load_code_from_local_folder = AsyncMock(
            return_value=[("def test():\n    pass", "test.py")]
        )

        await mock_testteller_agent.ingest_code_from_source(local_path)

        # Verify code loader was called
        mock_testteller_agent.code_loader.load_code_from_local_folder.assert_called_once_with(
            local_path)

        # Verify vector store was called
        mock_testteller_agent.vector_store.add_documents.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_ingest_code_from_source_no_cleanup(self, mock_testteller_agent):
        """Test ingesting code from remote repository without cleanup."""
        repo_url = "https://github.com/test/repo.git"

        # Mock code loader
        mock_testteller_agent.code_loader.load_code_from_repo = AsyncMock(
            return_value=[("def test():\n    pass", "test.py")]
        )
        mock_testteller_agent.code_loader.cleanup_repo = AsyncMock()

        await mock_testteller_agent.ingest_code_from_source(repo_url, cleanup_github_after=False)

        # Verify cleanup was not called
        mock_testteller_agent.code_loader.cleanup_repo.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_ingest_code_from_source_no_files(self, mock_testteller_agent):
        """Test ingesting code when no files are found."""
        repo_url = "https://github.com/test/empty-repo.git"

        # Mock code loader to return empty list
        mock_testteller_agent.code_loader.load_code_from_repo = AsyncMock(
            return_value=[])

        await mock_testteller_agent.ingest_code_from_source(repo_url)

        # Verify vector store was not called
        mock_testteller_agent.vector_store.add_documents.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_ingest_code_from_source_error(self, mock_testteller_agent):
        """Test error handling during code ingestion."""
        repo_url = "https://github.com/test/repo.git"

        # Mock code loader to raise exception
        mock_testteller_agent.code_loader.load_code_from_repo = AsyncMock(
            side_effect=Exception("Clone error")
        )

        with pytest.raises(Exception, match="Clone error"):
            await mock_testteller_agent.ingest_code_from_source(repo_url)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_get_ingested_data_count(self, mock_testteller_agent):
        """Test getting count of ingested documents."""
        mock_testteller_agent.vector_store.get_collection_count.return_value = 5

        count = await mock_testteller_agent.get_ingested_data_count()

        assert count == 5
        mock_testteller_agent.vector_store.get_collection_count.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_clear_ingested_data(self, mock_testteller_agent):
        """Test clearing all ingested data."""
        mock_testteller_agent.code_loader.cleanup_all_repos = AsyncMock()

        await mock_testteller_agent.clear_ingested_data()

        mock_testteller_agent.vector_store.clear_collection.assert_called_once()
        mock_testteller_agent.code_loader.cleanup_all_repos.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_clear_ingested_data_error(self, mock_testteller_agent):
        """Test error handling during data clearing."""
        mock_testteller_agent.vector_store.clear_collection.side_effect = Exception(
            "Clear error")

        with pytest.raises(Exception, match="Clear error"):
            await mock_testteller_agent.clear_ingested_data()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_generate_test_cases(self, mock_testteller_agent, mock_llm_response):
        """Test generating test cases."""
        code_context = "def login(username, password):\n    pass"

        # Mock vector store query
        mock_testteller_agent.vector_store.query_similar.return_value = {
            "documents": [["Sample test case 1", "Sample test case 2"]],
            "metadatas": [[{"source": "test1.py"}, {"source": "test2.py"}]],
            "distances": [[0.1, 0.2]]
        }

        # Mock LLM manager
        mock_testteller_agent.llm_manager.generate_text_async = AsyncMock(
            return_value=mock_llm_response)

        result = await mock_testteller_agent.generate_test_cases(code_context, n_retrieved_docs=2)

        assert result == mock_llm_response
        mock_testteller_agent.vector_store.query_similar.assert_called_once_with(
            query_text=code_context,
            n_results=2
        )
        mock_testteller_agent.llm_manager.generate_text_async.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_generate_test_cases_with_provider_optimization(self, mock_testteller_agent, mock_llm_response):
        """Test generating test cases with provider-specific optimization."""
        code_context = "def login(username, password):\n    pass"

        # Mock vector store query
        mock_testteller_agent.vector_store.query_similar.return_value = {
            "documents": [["Sample test case 1"]],
            "metadatas": [[{"source": "test1.py"}]],
            "distances": [[0.1]]
        }

        # Mock LLM manager to return different provider
        mock_testteller_agent.llm_manager.get_current_provider.return_value = "openai"
        mock_testteller_agent.llm_manager.generate_text_async = AsyncMock(
            return_value=mock_llm_response)

        with patch('testteller.agent.testteller_agent.get_test_case_generation_prompt') as mock_prompt:
            mock_prompt.return_value = "Optimized prompt"

            result = await mock_testteller_agent.generate_test_cases(code_context)

            assert result == mock_llm_response
            mock_prompt.assert_called_once()
            args = mock_prompt.call_args
            assert args[1]["provider"] == "openai"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_generate_test_cases_error(self, mock_testteller_agent):
        """Test error handling during test case generation."""
        code_context = "def login(username, password):\n    pass"

        # Mock vector store to raise exception
        mock_testteller_agent.vector_store.query_similar.side_effect = Exception(
            "Query error")

        with pytest.raises(Exception, match="Query error"):
            await mock_testteller_agent.generate_test_cases(code_context)

    @pytest.mark.unit
    def test_add_test_cases(self, mock_testteller_agent):
        """Test adding test cases to vector store."""
        test_cases = ["Test case 1", "Test case 2"]
        metadatas = [{"source": "test1"}, {"source": "test2"}]
        ids = ["id1", "id2"]

        mock_testteller_agent.add_test_cases(test_cases, metadatas, ids)

        mock_testteller_agent.vector_store.add_documents.assert_called_once_with(
            documents=test_cases,
            metadatas=metadatas,
            ids=ids
        )

    @pytest.mark.unit
    def test_add_test_cases_error(self, mock_testteller_agent):
        """Test error handling when adding test cases."""
        test_cases = ["Test case 1"]
        mock_testteller_agent.vector_store.add_documents.side_effect = Exception(
            "Add error")

        with pytest.raises(Exception, match="Add error"):
            mock_testteller_agent.add_test_cases(test_cases)

    @pytest.mark.unit
    def test_clear_test_cases(self, mock_testteller_agent):
        """Test clearing test cases from vector store."""
        mock_testteller_agent.clear_test_cases()

        mock_testteller_agent.vector_store.clear_collection.assert_called_once()

    @pytest.mark.unit
    def test_clear_test_cases_error(self, mock_testteller_agent):
        """Test error handling when clearing test cases."""
        mock_testteller_agent.vector_store.clear_collection.side_effect = Exception(
            "Clear error")

        with pytest.raises(Exception, match="Clear error"):
            mock_testteller_agent.clear_test_cases()

    @pytest.mark.unit
    def test_backward_compatibility_alias(self):
        """Test that TestTellerRagAgent alias exists for backward compatibility."""
        from testteller.agent.testteller_agent import TestTellerRagAgent
        assert TestTellerRagAgent == TestTellerAgent
