import pytest
import asyncio
from unittest import mock
import logging
import os
from pathlib import Path

# Class to be tested
from data_ingestion.text_splitter import TestTellerAgent # Agent from text_splitter.py

# Mocks for dependencies (similar to tests/test_agent.py)
@pytest.fixture
def mock_gemini_client_ts(mocker): # Suffix _ts for text_splitter
    mock_client = mocker.MagicMock(name="MockGeminiClientForTSAgent")
    mock_client.generate_text_async = mocker.AsyncMock(return_value="mocked gemini text for ts agent")
    # Add other methods if TestTellerAgent uses them directly
    return mock_client

@pytest.fixture
def mock_vector_store_ts(mocker):
    mock_store = mocker.MagicMock(name="MockChromaDBManagerForTSAgent")
    mock_store.add_documents = mocker.AsyncMock() # Renamed from add_documents_async
    mock_store.query_collection = mocker.AsyncMock(return_value=[]) # Renamed from query_collection_async
    mock_store.get_collection_count_async = mocker.AsyncMock(return_value=0)
    mock_store.clear_collection_async = mocker.AsyncMock()
    mock_store.generate_id_from_text_and_source = mocker.MagicMock(return_value="mock_doc_id_ts")
    mock_store.db_path = "mock/db/path_ts"
    return mock_store

@pytest.fixture
def mock_document_loader_ts(mocker):
    mock_loader = mocker.MagicMock(name="MockDocumentLoaderForTSAgent")
    mock_loader.load_document = mocker.AsyncMock(return_value="mock file content ts")
    mock_loader.load_from_directory = mocker.AsyncMock(return_value=[("doc_path_ts1", "doc_content_ts1")])
    return mock_loader

@pytest.fixture
def mock_code_loader_ts(mocker):
    mock_loader = mocker.MagicMock(name="MockCodeLoaderForTSAgent")
    mock_loader.load_code_from_repo = mocker.AsyncMock(return_value=[("repo/file_ts.py", "repo code ts")])
    mock_loader.cleanup_repo = mocker.AsyncMock()
    mock_loader.cleanup_all_repos = mocker.AsyncMock()
    # Add load_code_from_local_folder if TestTellerAgent uses it
    return mock_loader

@pytest.fixture
def mock_text_splitter_ts(mocker): # Mock the TextSplitter used by TestTellerAgent
    mock_splitter = mocker.MagicMock(name="MockTextSplitterForTSAgent")
    mock_splitter.split_text = mocker.MagicMock(return_value=["chunk_ts1", "chunk_ts2"])
    return mock_splitter

@pytest.fixture
def ts_agent_fixture(mocker, mock_gemini_client_ts, mock_vector_store_ts, mock_document_loader_ts, mock_code_loader_ts, mock_text_splitter_ts):
    # Patch constructors in the 'data_ingestion.text_splitter' module's namespace and capture mocks
    patched_gemini_client_constructor = mocker.patch("data_ingestion.text_splitter.GeminiClient", return_value=mock_gemini_client_ts)
    patched_chromadb_manager_constructor = mocker.patch("data_ingestion.text_splitter.ChromaDBManager", return_value=mock_vector_store_ts)
    patched_document_loader_constructor = mocker.patch("data_ingestion.text_splitter.DocumentLoader", return_value=mock_document_loader_ts)
    patched_code_loader_constructor = mocker.patch("data_ingestion.text_splitter.CodeLoader", return_value=mock_code_loader_ts)
    patched_text_splitter_constructor = mocker.patch("data_ingestion.text_splitter.TextSplitter", return_value=mock_text_splitter_ts)

    # Mock settings directly within the data_ingestion.text_splitter module
    mock_settings = mocker.patch("data_ingestion.text_splitter.settings")
    mock_settings.chroma_db.default_collection_name = "test_ts_collection"

    agent = TestTellerAgent(collection_name="test_ts_agent_collection")

    return {
        "agent": agent,
        "gemini_constructor": patched_gemini_client_constructor,
        "chromadb_constructor": patched_chromadb_manager_constructor,
        "doc_loader_constructor": patched_document_loader_constructor,
        "code_loader_constructor": patched_code_loader_constructor,
        "text_splitter_constructor": patched_text_splitter_constructor,
        # Pass through actual mock instances for validating 'with' arguments if needed elsewhere
        "mock_gemini_client_ts": mock_gemini_client_ts,
        "mock_vector_store_ts": mock_vector_store_ts,
        "mock_text_splitter_ts": mock_text_splitter_ts,
    }

# --- Test __init__ ---
# Note: Removed unused mock_document_loader_ts, mock_code_loader_ts, mock_text_splitter_ts from direct args
# as they are indirectly used via the constructors or available in fixture_data if needed.
# mocker is also not directly needed here anymore.
def test_ts_agent_init(ts_agent_fixture):
    fixture_data = ts_agent_fixture
    agent = fixture_data["agent"]
    mock_gemini_client_ts_instance = fixture_data["mock_gemini_client_ts"] # Used for `with` check

    # Check that constructors of dependencies were called
    fixture_data["gemini_constructor"].assert_called_once()
    fixture_data["chromadb_constructor"].assert_called_once_with(
        gemini_client=mock_gemini_client_ts_instance, collection_name="test_ts_agent_collection"
    )
    fixture_data["doc_loader_constructor"].assert_called_once()
    fixture_data["code_loader_constructor"].assert_called_once()
    fixture_data["text_splitter_constructor"].assert_called_once()

    assert agent.collection_name == "test_ts_agent_collection"

# --- Test _ingest_content ---
@pytest.mark.asyncio
async def test_ts_agent_ingest_content_success(ts_agent_fixture, caplog):
    agent = ts_agent_fixture["agent"]
    mock_vector_store_ts = ts_agent_fixture["mock_vector_store_ts"]
    mock_text_splitter_ts = ts_agent_fixture["mock_text_splitter_ts"]
    caplog.set_level(logging.INFO)
    contents = [("file1.py", "code1"), ("file2.txt", "text1")]
    source_type = "test_src"

    # Configure side effect for split_text if it varies per content
    mock_text_splitter_ts.split_text.side_effect = lambda text: [f"chunk_{text}_1", f"chunk_{text}_2"]

    # Configure side effect for id generation
    id_counter = 0
    def id_gen_side_effect(text, source):
        nonlocal id_counter
        id_counter += 1
        return f"id_ts_{id_counter}"
    mock_vector_store_ts.generate_id_from_text_and_source.side_effect = id_gen_side_effect

    await agent._ingest_content(contents, source_type)

    mock_text_splitter_ts.split_text.assert_any_call("code1")
    mock_text_splitter_ts.split_text.assert_any_call("text1")

    mock_vector_store_ts.add_documents.assert_called_once()
    call_args = mock_vector_store_ts.add_documents.call_args
    call_kwargs = call_args[1] if call_args and len(call_args) > 1 else call_args[0][0] if call_args and call_args[0] else {}


    assert call_kwargs['documents'] == ["chunk_code1_1", "chunk_code1_2", "chunk_text1_1", "chunk_text1_2"]
    assert call_kwargs['ids'] == ["id_ts_1", "id_ts_2", "id_ts_3", "id_ts_4"]
    assert len(call_kwargs['metadatas']) == 4
    assert call_kwargs['metadatas'][0]['source'] == "file1.py"
    assert "Content preparation for 4 chunks from 2 sources took" in caplog.text

@pytest.mark.asyncio
async def test_ts_agent_ingest_content_with_empty_item_content(ts_agent_fixture, caplog):
    agent = ts_agent_fixture["agent"]
    mock_vector_store_ts = ts_agent_fixture["mock_vector_store_ts"]
    mock_text_splitter_ts = ts_agent_fixture["mock_text_splitter_ts"]
    caplog.set_level(logging.INFO) # To capture INFO and WARNING

    contents = [
        ("file1.py", "code1"),
        ("file2.txt", ""), # Empty content
        ("file3.md", "  "), # Whitespace only content
        ("file4.py", "code2")
    ]
    source_type = "mixed_content_src"

    # Configure side effect for split_text
    def split_text_side_effect(text):
        if text == "code1":
            return ["chunk_code1_1"]
        elif text == "code2":
            return ["chunk_code2_1"]
        return []
    mock_text_splitter_ts.split_text.side_effect = split_text_side_effect

    # Configure side effect for id generation
    id_counter = 0
    def id_gen_side_effect(text, source):
        nonlocal id_counter
        id_counter += 1
        return f"id_mixed_{id_counter}"
    mock_vector_store_ts.generate_id_from_text_and_source.side_effect = id_gen_side_effect

    await agent._ingest_content(contents, source_type)

    # split_text should only be called for non-empty, non-whitespace content
    mock_text_splitter_ts.split_text.assert_any_call("code1")
    mock_text_splitter_ts.split_text.assert_any_call("code2")
    assert mock_text_splitter_ts.split_text.call_count == 2

    mock_vector_store_ts.add_documents.assert_called_once()
    call_args = mock_vector_store_ts.add_documents.call_args
    call_kwargs = call_args[1] if call_args and len(call_args) > 1 else call_args[0][0] if call_args and call_args[0] else {}

    assert call_kwargs['documents'] == ["chunk_code1_1", "chunk_code2_1"]
    assert call_kwargs['ids'] == ["id_mixed_1", "id_mixed_2"]
    assert len(call_kwargs['metadatas']) == 2
    assert call_kwargs['metadatas'][0]['source'] == "file1.py"
    assert call_kwargs['metadatas'][1]['source'] == "file4.py"

    assert "Skipping empty content from file2.txt" in caplog.text
    assert "Skipping empty content from file3.md" in caplog.text
    assert "Content preparation for 2 chunks from 4 sources took" in caplog.text


@pytest.mark.asyncio
async def test_ts_agent_ingest_content_empty_input(ts_agent_fixture, caplog):
    agent = ts_agent_fixture["agent"]
    mock_vector_store_ts = ts_agent_fixture["mock_vector_store_ts"]
    mock_text_splitter_ts = ts_agent_fixture["mock_text_splitter_ts"]
    caplog.set_level(logging.INFO)

    await agent._ingest_content([], "empty_src")

    mock_text_splitter_ts.split_text.assert_not_called()
    mock_vector_store_ts.add_documents.assert_not_called()
    assert "No content provided for ingestion from empty_src." in caplog.text # Updated log message

@pytest.mark.asyncio
async def test_ts_agent_ingest_content_no_chunks_produced(ts_agent_fixture, caplog):
    agent = ts_agent_fixture["agent"]
    mock_vector_store_ts = ts_agent_fixture["mock_vector_store_ts"]
    mock_text_splitter_ts = ts_agent_fixture["mock_text_splitter_ts"]
    caplog.set_level(logging.INFO)

    contents = [("file1.txt", "some text")]
    source_type = "no_chunk_src"

    mock_text_splitter_ts.split_text.return_value = [] # Simulate no chunks

    await agent._ingest_content(contents, source_type)

    mock_text_splitter_ts.split_text.assert_called_once_with("some text")
    mock_vector_store_ts.add_documents.assert_not_called()
    # Check for the aggregate message, not the per-file warning for this specific test's intent
    assert "No valid chunks to ingest from no_chunk_src." in caplog.text # Updated log message


# --- Add more tests for other methods of TestTellerAgent ---

# --- Tests for ingest_documents_from_path ---
@pytest.mark.asyncio
async def test_ts_agent_ingest_documents_from_path_single_file_success(ts_agent_fixture, mocker, caplog):
    agent = ts_agent_fixture["agent"]
    mock_document_loader_ts = ts_agent_fixture["agent"].document_loader # Get from agent instance
    caplog.set_level(logging.INFO)

    mocker.patch.object(agent, "_ingest_content", mocker.AsyncMock())

    # Mock os.path.isfile and os.path.isdir as used in the agent
    mocker.patch("data_ingestion.text_splitter.os.path.isfile", return_value=True)
    mocker.patch("data_ingestion.text_splitter.os.path.isdir", return_value=False)

    doc_content = "content from single file"
    mock_document_loader_ts.load_document.return_value = doc_content
    path_arg = "dummy/file.txt"

    await agent.ingest_documents_from_path(path_arg)

    mock_document_loader_ts.load_document.assert_called_once_with(path_arg)
    agent._ingest_content.assert_called_once_with([(path_arg, doc_content)], source_type="document_file")
    assert f"Document ingestion from path '{path_arg}' completed in" in caplog.text

@pytest.mark.asyncio
async def test_ts_agent_ingest_documents_from_path_directory_success(ts_agent_fixture, mocker, caplog):
    agent = ts_agent_fixture["agent"]
    mock_document_loader_ts = agent.document_loader
    caplog.set_level(logging.INFO)

    mocker.patch.object(agent, "_ingest_content", mocker.AsyncMock())

    mocker.patch("data_ingestion.text_splitter.os.path.isfile", return_value=False)
    mocker.patch("data_ingestion.text_splitter.os.path.isdir", return_value=True)

    loaded_docs = [("doc1.txt", "content1"), ("doc2.pdf", "content2")]
    mock_document_loader_ts.load_from_directory.return_value = loaded_docs
    path_arg = "dummy/docs_dir"

    await agent.ingest_documents_from_path(path_arg)

    mock_document_loader_ts.load_from_directory.assert_called_once_with(path_arg)
    agent._ingest_content.assert_called_once_with(loaded_docs, source_type="document_directory")
    assert f"Document ingestion from path '{path_arg}' completed in" in caplog.text

@pytest.mark.asyncio
async def test_ts_agent_ingest_documents_from_path_non_existent(ts_agent_fixture, mocker, caplog):
    agent = ts_agent_fixture["agent"]
    mock_document_loader_ts = agent.document_loader
    caplog.set_level(logging.ERROR) # Expecting an error log

    mocker.patch.object(agent, "_ingest_content", mocker.AsyncMock())

    mocker.patch("data_ingestion.text_splitter.os.path.isfile", return_value=False)
    mocker.patch("data_ingestion.text_splitter.os.path.isdir", return_value=False) # Neither a file nor a directory
    path_arg = "non_existent/path"

    await agent.ingest_documents_from_path(path_arg)

    mock_document_loader_ts.load_document.assert_not_called()
    mock_document_loader_ts.load_from_directory.assert_not_called()
    agent._ingest_content.assert_not_called()
    assert f"Path does not exist or is not a file/directory: {path_arg}" in caplog.text

@pytest.mark.asyncio
async def test_ts_agent_ingest_documents_from_path_single_file_no_content(ts_agent_fixture, mocker, caplog):
    agent = ts_agent_fixture["agent"]
    mock_document_loader_ts = agent.document_loader
    caplog.set_level(logging.WARNING)

    mocker.patch.object(agent, "_ingest_content", mocker.AsyncMock())

    mocker.patch("data_ingestion.text_splitter.os.path.isfile", return_value=True)
    mocker.patch("data_ingestion.text_splitter.os.path.isdir", return_value=False)

    mock_document_loader_ts.load_document.return_value = None # Simulate no content loaded
    path_arg = "dummy/empty_file.txt"

    await agent.ingest_documents_from_path(path_arg)

    mock_document_loader_ts.load_document.assert_called_once_with(path_arg)
    agent._ingest_content.assert_not_called()
    assert f"Could not load document content from file: {path_arg}" in caplog.text

@pytest.mark.asyncio
async def test_ts_agent_ingest_documents_from_path_directory_no_content(ts_agent_fixture, mocker, caplog):
    agent = ts_agent_fixture["agent"]
    mock_document_loader_ts = agent.document_loader
    caplog.set_level(logging.WARNING)

    mocker.patch.object(agent, "_ingest_content", mocker.AsyncMock())

    mocker.patch("data_ingestion.text_splitter.os.path.isfile", return_value=False)
    mocker.patch("data_ingestion.text_splitter.os.path.isdir", return_value=True)

    mock_document_loader_ts.load_from_directory.return_value = [] # Simulate no documents loaded
    path_arg = "dummy/empty_dir"

    await agent.ingest_documents_from_path(path_arg)

    mock_document_loader_ts.load_from_directory.assert_called_once_with(path_arg)
    agent._ingest_content.assert_not_called()
    assert f"No documents loaded from directory: {path_arg}" in caplog.text


# --- Tests for ingest_code_from_github ---
@pytest.mark.asyncio
async def test_ts_agent_ingest_code_from_github_success_with_cleanup(ts_agent_fixture, mocker, caplog):
    agent = ts_agent_fixture["agent"]
    mock_code_loader_ts = agent.code_loader # Get from agent instance
    caplog.set_level(logging.INFO)

    mocker.patch.object(agent, "_ingest_content", mocker.AsyncMock())

    repo_url = "https://github.com/example/repo"
    loaded_code = [("file1.py", "code content 1"), ("file2.js", "code content 2")]
    mock_code_loader_ts.load_code_from_repo.return_value = loaded_code

    await agent.ingest_code_from_github(repo_url) # cleanup_after defaults to True

    mock_code_loader_ts.load_code_from_repo.assert_called_once_with(repo_url)
    agent._ingest_content.assert_called_once_with(loaded_code, source_type="github_code")
    mock_code_loader_ts.cleanup_repo.assert_called_once_with(repo_url)
    assert f"Code ingestion from repo '{repo_url}' completed in" in caplog.text

@pytest.mark.asyncio
async def test_ts_agent_ingest_code_from_github_success_no_cleanup(ts_agent_fixture, mocker, caplog):
    agent = ts_agent_fixture["agent"]
    mock_code_loader_ts = agent.code_loader
    caplog.set_level(logging.INFO)

    mocker.patch.object(agent, "_ingest_content", mocker.AsyncMock())

    repo_url = "https://github.com/example/repo_no_cleanup"
    loaded_code = [("file.py", "code")]
    mock_code_loader_ts.load_code_from_repo.return_value = loaded_code

    await agent.ingest_code_from_github(repo_url, cleanup_after=False)

    mock_code_loader_ts.load_code_from_repo.assert_called_once_with(repo_url)
    agent._ingest_content.assert_called_once_with(loaded_code, source_type="github_code")
    mock_code_loader_ts.cleanup_repo.assert_not_called()
    assert f"Code ingestion from repo '{repo_url}' completed in" in caplog.text

@pytest.mark.asyncio
async def test_ts_agent_ingest_code_from_github_no_content_loaded(ts_agent_fixture, mocker, caplog):
    agent = ts_agent_fixture["agent"]
    mock_code_loader_ts = agent.code_loader
    caplog.set_level(logging.INFO) # Changed to INFO to capture both WARNING and INFO logs

    mocker.patch.object(agent, "_ingest_content", mocker.AsyncMock())

    repo_url = "https://github.com/example/empty_repo"
    mock_code_loader_ts.load_code_from_repo.return_value = [] # Simulate no code loaded

    await agent.ingest_code_from_github(repo_url) # cleanup_after defaults to True

    mock_code_loader_ts.load_code_from_repo.assert_called_once_with(repo_url)
    agent._ingest_content.assert_not_called()
    mock_code_loader_ts.cleanup_repo.assert_called_once_with(repo_url) # Cleanup should still occur
    assert f"No code files loaded from GitHub repo: {repo_url}" in caplog.text
    assert f"Code ingestion from repo '{repo_url}' completed in" in caplog.text # Completion message still logged


# --- Test for get_ingested_data_count ---
@pytest.mark.asyncio
async def test_ts_agent_get_ingested_data_count(ts_agent_fixture):
    agent = ts_agent_fixture["agent"]
    mock_vector_store_ts = ts_agent_fixture["mock_vector_store_ts"] # Get the mock directly

    expected_count = 42
    mock_vector_store_ts.get_collection_count_async.return_value = expected_count

    actual_count = await agent.get_ingested_data_count()

    mock_vector_store_ts.get_collection_count_async.assert_called_once()
    assert actual_count == expected_count

# --- Test for clear_ingested_data ---
@pytest.mark.asyncio
async def test_ts_agent_clear_ingested_data(ts_agent_fixture, caplog):
    agent = ts_agent_fixture["agent"]
    mock_vector_store_ts = ts_agent_fixture["mock_vector_store_ts"]
    mock_code_loader_ts = agent.code_loader # Get from agent instance
    caplog.set_level(logging.INFO)

    collection_name = agent.collection_name

    await agent.clear_ingested_data()

    mock_vector_store_ts.clear_collection_async.assert_called_once()
    mock_code_loader_ts.cleanup_all_repos.assert_called_once() # This is part of TestTellerAgent
    assert f"Clearing all ingested data from collection '{collection_name}'." in caplog.text
    assert f"Data cleared for collection '{collection_name}' in" in caplog.text


# --- Tests for generate_test_cases ---
@pytest.mark.asyncio
async def test_ts_agent_generate_test_cases_with_context(ts_agent_fixture, mocker, caplog):
    agent = ts_agent_fixture["agent"]
    mock_vector_store_ts = ts_agent_fixture["mock_vector_store_ts"]
    mock_gemini_client_ts = ts_agent_fixture["mock_gemini_client_ts"] # Get from fixture data
    caplog.set_level(logging.INFO)

    mocker.patch.object(agent, "get_ingested_data_count", return_value=10) # Mock this method on agent

    query = "test query with context"
    retrieved_docs_data = [
        {'document': 'doc content 1', 'metadata': {'source': 'source1'}, 'distance': 0.1},
        {'document': 'doc content 2', 'metadata': {'source': 'source2'}, 'distance': 0.2}
    ]
    mock_vector_store_ts.query_collection.return_value = retrieved_docs_data

    expected_response = "Generated test cases based on context."
    mock_gemini_client_ts.generate_text_async.return_value = expected_response

    response = await agent.generate_test_cases(query)

    agent.get_ingested_data_count.assert_called_once()
    mock_vector_store_ts.query_collection.assert_called_once_with(query_text=query, n_results=5)
    mock_gemini_client_ts.generate_text_async.assert_called_once()
    # Check that context from docs is in the prompt passed to Gemini
    prompt_arg = mock_gemini_client_ts.generate_text_async.call_args[0][0]
    assert "doc content 1" in prompt_arg
    assert "doc content 2" in prompt_arg
    assert "Source: source1" in prompt_arg
    assert response == expected_response
    assert f"Generating test cases for query: '{query}'" in caplog.text
    assert "Context retrieval took" in caplog.text
    assert "LLM generation took" in caplog.text

@pytest.mark.asyncio
async def test_ts_agent_generate_test_cases_empty_kb(ts_agent_fixture, mocker, caplog):
    agent = ts_agent_fixture["agent"]
    mock_vector_store_ts = ts_agent_fixture["mock_vector_store_ts"]
    mock_gemini_client_ts = ts_agent_fixture["mock_gemini_client_ts"]
    caplog.set_level(logging.INFO)

    mocker.patch.object(agent, "get_ingested_data_count", return_value=0) # Empty KB

    query = "test query empty kb"
    expected_response = "Generated test cases for empty KB."
    mock_gemini_client_ts.generate_text_async.return_value = expected_response

    response = await agent.generate_test_cases(query)

    agent.get_ingested_data_count.assert_called_once()
    mock_vector_store_ts.query_collection.assert_not_called() # Should not be called if KB is empty
    mock_gemini_client_ts.generate_text_async.assert_called_once()
    prompt_arg = mock_gemini_client_ts.generate_text_async.call_args[0][0]
    assert "No specific context documents were found" in prompt_arg
    assert response == expected_response
    assert f"No data ingested in collection '{agent.collection_name}'. Test case generation might be suboptimal." in caplog.text

@pytest.mark.asyncio
async def test_ts_agent_generate_test_cases_no_relevant_docs(ts_agent_fixture, mocker, caplog):
    agent = ts_agent_fixture["agent"]
    mock_vector_store_ts = ts_agent_fixture["mock_vector_store_ts"]
    mock_gemini_client_ts = ts_agent_fixture["mock_gemini_client_ts"]
    caplog.set_level(logging.INFO)

    mocker.patch.object(agent, "get_ingested_data_count", return_value=10) # KB has data
    mock_vector_store_ts.query_collection.return_value = [] # But no docs for this query

    query = "test query no relevant docs"
    expected_response = "Generated test cases for no relevant docs."
    mock_gemini_client_ts.generate_text_async.return_value = expected_response

    response = await agent.generate_test_cases(query)

    agent.get_ingested_data_count.assert_called_once()
    mock_vector_store_ts.query_collection.assert_called_once_with(query_text=query, n_results=5)
    mock_gemini_client_ts.generate_text_async.assert_called_once()
    prompt_arg = mock_gemini_client_ts.generate_text_async.call_args[0][0]
    assert "No relevant context documents were found" in prompt_arg
    assert response == expected_response
    assert f"No relevant documents found for query: '{query}'" in caplog.text


# Placeholder for a test to ensure the file runs
def test_placeholder_ts_agent():
    assert True
