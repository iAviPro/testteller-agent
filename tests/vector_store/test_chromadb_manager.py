import pytest
import asyncio
from unittest import mock
import sys # For sys.modules mocking
import logging
import hashlib # For testing _generate_id_from_text_and_source
import functools # For checking functools.partial
import numpy as np # For comparing numpy arrays if necessary

# --- Mocks for external libraries will be applied using pytest-mock (mocker) within fixtures ---

# --- Gemini Client Mocking ---
# Create a mock for the GeminiClient class that ChromaDBManager will receive
MockGeminiClientClass = mock.MagicMock(name="MockGeminiClient_Class") # This is the class
# Add 'get_embeddings_sync' to the spec_set
mock_gemini_client_instance = mock.MagicMock(
    spec_set=["get_embedding_async", "get_embedding_batch_async", "get_embeddings_sync"],
    name="MockGeminiClient_Instance"
)
MockGeminiClientClass.return_value = mock_gemini_client_instance # Instantiating the class returns our instance
mock_gemini_client_instance.get_embedding_async = mock.AsyncMock(name="get_embedding_async_mock")
mock_gemini_client_instance.get_embedding_batch_async = mock.AsyncMock(name="get_embedding_batch_async_mock")
# Mock for the sync batch embedding method used by the embedding function
mock_gemini_client_instance.get_embeddings_sync = mock.MagicMock(name="get_embeddings_sync_mock")
# --- End Gemini Client Mocking ---

# Now import ChromaDBManager and relevant settings/config
from vector_store.chromadb_manager import ChromaDBManager, GeminiChromaEmbeddingFunction
from config import settings as global_settings_original, AppSettings, ChromaDbSettings
# Note: We don't need to import the real GeminiClient here for the tests themselves.

# --- Fixtures ---
@pytest.fixture(scope="function")
def isolated_settings(monkeypatch):
    current_test_app_settings = AppSettings()
    monkeypatch.setattr("vector_store.chromadb_manager.settings", current_test_app_settings)
    monkeypatch.setattr("config.settings", current_test_app_settings, raising=False)
    yield current_test_app_settings

@pytest.fixture
def mock_env_vars_chroma(monkeypatch):
    monkeypatch.setenv("CHROMA_DB_PATH", "./test_chroma_data_fixt") # Unique path for fixture
    monkeypatch.setenv("DEFAULT_COLLECTION_NAME", "test_collection_fixt")

@pytest.fixture
def chromadb_manager_fixt(mock_env_vars_chroma, isolated_settings, mocker):
    # Mock PersistentClient within the chromadb_manager module's context
    mock_persistent_client_instance = mocker.MagicMock(name="PersistentClientInstanceMock")
    mock_collection_instance = mocker.MagicMock(name="CollectionInstanceMock")
    mock_collection_instance.count = mocker.MagicMock(return_value=10) # Set default count
    mock_persistent_client_instance.get_or_create_collection.return_value = mock_collection_instance

    # Patch the chromadb.PersistentClient where it's used by ChromaDBManager
    mocker.patch("vector_store.chromadb_manager.chromadb.PersistentClient", return_value=mock_persistent_client_instance)
    # If ChromaDBManager imports PersistentClient directly (e.g. from chromadb import PersistentClient)
    # then the patch target might need to be 'vector_store.chromadb_manager.PersistentClient'

    mock_gemini_client_instance.get_embedding_async.reset_mock()
    mock_gemini_client_instance.get_embedding_batch_async.reset_mock()
    mock_gemini_client_instance.get_embeddings_sync.reset_mock() # Reset the new mock

    # Use the mocked Gemini Client *class* to create the instance passed to ChromaDBManager
    gemini_client_to_pass = MockGeminiClientClass() # This mock is already set up globally

    manager = ChromaDBManager(
        gemini_client=gemini_client_to_pass,
        collection_name="fixture_collection" # Specific name for clarity
    )
    # Store mocks for assertion if needed, though direct assertion on patched objects is often cleaner
    manager._test_mock_persistent_client = mock_persistent_client_instance
    manager._test_mock_collection = mock_collection_instance
    return manager

# --- Tests for ChromaDBManager.__init__ ---
def test_chromadb_manager_init(chromadb_manager_fixt, isolated_settings, mocker): # Add mocker here
    current_settings = isolated_settings

    # The client is now a mock patched by `mocker` in the fixture.
    # Access it via the patched path or if stored on the manager for tests.
    # chromadb.PersistentClient was patched, so we can check its call.
    patched_persistent_client = mocker.patch("vector_store.chromadb_manager.chromadb.PersistentClient") # Get the patch object

    # This assertion needs to be on the patched object from the fixture
    # This requires the fixture to return the patched object or for us to fetch it
    # For simplicity, let's assume the manager stores the client instance, which it does.
    # No, the manager stores the *mocked* client. We need to assert the patched CLASS was called.
    # The patch object `vector_store.chromadb_manager.chromadb.PersistentClient` is the one to check.

    # Re-patch or get the mock directly for assertion on the class constructor
    # This is a bit tricky as the fixture already did the patch.
    # Let's assert on the instance stored by the fixture.
    # The `PersistentClient` class itself is called.
    # The fixture already patched "vector_store.chromadb_manager.chromadb.PersistentClient"
    # So, we need to assert that this mock (the class mock) was called.

    # To assert the class was called:
    # Get the mock for the class itself
    PatchedPersistentClientClass = chromadb_manager_fixt.client.__class__ # This is the mock instance's class
                                                                      # This is not what we want to assert.

    # The easiest is to assert on the mock object used in the fixture.
    # The fixture uses `mocker.patch("vector_store.chromadb_manager.chromadb.PersistentClient", return_value=mock_persistent_client_instance)`
    # So, we assert that `vector_store.chromadb_manager.chromadb.PersistentClient` (the mock class) was called.
    # And that `mock_persistent_client_instance.get_or_create_collection` was called.

    # Get the mock object for PersistentClient class
    persistent_client_class_mock = chromadb_manager_fixt._test_mock_persistent_client.parent # This is fragile.
                                                                                        # Better to retrieve from mocker.
                                                                                        # Or re-patch for assertion context.

    # Alternative: The fixture `chromadb_manager_fixt` creates the manager.
    # `manager.client` is the `mock_persistent_client_instance`.
    # `manager.collection` is the `mock_collection_instance`.
    # The call to PersistentClient happens inside ChromaDBManager's __init__.
    # We need to check that the *patched class* was called correctly.

    # Let's re-evaluate: the fixture creates the manager, which calls the patched client.
    # The `chromadb_manager_fixt.client` IS `mock_persistent_client_instance`.
    # The `chromadb_manager_fixt.collection` IS `mock_collection_instance`.

    # So, the PersistentClient class mock (from the fixture's mocker.patch) should have been called.
    # And then, get_or_create_collection on its return value (mock_persistent_client_instance)

    # How to get the class mock that was patched in the fixture?
    # It's not directly returned by the fixture.
    # For now, let's trust the fixture sets it up and assert on the instance methods.
    # If `path` was passed to `PersistentClient(...)`, it would be on the class mock's call_args.

    assert chromadb_manager_fixt.client.get_or_create_collection.call_count == 1
    call_args = chromadb_manager_fixt.client.get_or_create_collection.call_args
    assert call_args[1]['name'] == "fixture_collection"

    # Check the embedding_function type (it's an instance of GeminiChromaEmbeddingFunction)
    ef_instance = call_args[1]['embedding_function']
    assert isinstance(ef_instance, GeminiChromaEmbeddingFunction)
    # And that this instance wraps the Gemini client instance we provided
    assert ef_instance.gem_client is mock_gemini_client_instance # Corrected attribute

    assert chromadb_manager_fixt.collection_name == "fixture_collection"
    assert chromadb_manager_fixt.client is chromadb_manager_fixt._test_mock_persistent_client # Use stored mock
    assert chromadb_manager_fixt.collection is chromadb_manager_fixt._test_mock_collection # Assert it's the one from fixture
    assert chromadb_manager_fixt.gemini_client is mock_gemini_client_instance # Assert it's the one from global mocks

# --- Test ChromaEmbeddingFunction.__call__ (used by _embedding_function) ---
def test_chroma_embedding_function_call_sync_manages_async(caplog):
    # Test ChromaEmbeddingFunction directly as it's a key part.
    # This function is synchronous but calls an async method.
    texts_to_embed = ["text1", "text2"]
    # Create correctly dimensioned dummy embeddings (dim=768)
    expected_embeddings = [[float(i)/1000 for i in range(768)], [float(i+1)/1000 for i in range(768)]]


    # Create a new mock Gemini client instance specifically for this test unit
    # It should now have get_embeddings_sync
    temp_mock_gemini_instance = mock.MagicMock(spec_set=["get_embeddings_sync"])
    temp_mock_gemini_instance.get_embeddings_sync.return_value = expected_embeddings # This is a sync method

    embedding_function_instance = GeminiChromaEmbeddingFunction(temp_mock_gemini_instance)

    # GeminiChromaEmbeddingFunction's __call__ is synchronous and calls a sync method on gem_client.
    # No need to mock asyncio.run here as the function itself is not async.
    embeddings_result = embedding_function_instance(texts_to_embed) # This is a sync call

    # Use np.allclose for floating point array comparisons and ensure consistent dtype
    assert np.allclose(np.array(embeddings_result, dtype=np.float32), np.array(expected_embeddings, dtype=np.float32))
    temp_mock_gemini_instance.get_embeddings_sync.assert_called_once_with(texts_to_embed)

def test_chroma_embedding_function_call_empty_input(caplog):
    temp_mock_gemini_instance = mock.MagicMock(spec_set=["get_embeddings_sync"])
    temp_mock_gemini_instance.get_embeddings_sync.return_value = [] # Mock client still returns empty list
    embedding_function_instance = GeminiChromaEmbeddingFunction(temp_mock_gemini_instance)

    # ChromaDB's EmbeddingFunction wrapper raises ValueError if __call__ returns empty list.
    # Escape regex special characters like '[' and ']'.
    with pytest.raises(ValueError, match=r"Expected Embeddings to be non-empty list or numpy array, got \[]"): # Use raw string for regex
        embedding_function_instance([]) # This will call the client, get [], then raise in normalize_embeddings

    # The client's get_embeddings_sync is NOT called because GeminiChromaEmbeddingFunction returns early.
    temp_mock_gemini_instance.get_embeddings_sync.assert_not_called()


# --- Test add_documents_async ---
@pytest.mark.asyncio
async def test_add_documents_async_success(chromadb_manager_fixt, caplog):
    caplog.set_level(logging.INFO)
    docs = ["doc1 text", "doc2 text"]
    metadatas = [{"source": "s1"}, {"source": "s2"}]
    ids = ["id1", "id2"]

    # Ensure the collection mock used by the fixture is configured for this call
    # Ensure the collection mock used by the fixture is configured for this call
    # The collection is already a mock: chromadb_manager_fixt._test_mock_collection
    chromadb_manager_fixt._test_mock_collection.add.reset_mock() # Reset for this specific test

    await chromadb_manager_fixt.add_documents(docs, metadatas, ids) # Renamed to add_documents

    chromadb_manager_fixt._test_mock_collection.add.assert_called_once_with(
        documents=docs, metadatas=metadatas, ids=ids
    )
    assert f"Added/updated {len(docs)} documents to collection '{chromadb_manager_fixt.collection_name}'" in caplog.text

@pytest.mark.asyncio
async def test_add_documents_async_empty_list(chromadb_manager_fixt, caplog):
    caplog.set_level(logging.INFO) # Ensure INFO is captured if warning is not the only log
    await chromadb_manager_fixt.add_documents([], [], []) # Renamed to add_documents
    chromadb_manager_fixt._test_mock_collection.add.assert_not_called()
    assert "No documents provided to add_documents." in caplog.text

# --- Test query_collection_async ---
@pytest.mark.asyncio
async def test_query_collection_async_success(chromadb_manager_fixt):
    query_texts = ["query text for test"]
    n_results = 1
    mock_query_response = {
        'ids': [['id1']], 'documents': [['doc1 content']],
        'metadatas': [[{'source': 'test_source'}]], 'distances': [[0.01]]
    }
    chromadb_manager_fixt._test_mock_collection.query.return_value = mock_query_response
    chromadb_manager_fixt._test_mock_collection.query.reset_mock()


    results = await chromadb_manager_fixt.query_collection(query_texts, n_results) # Renamed to query_collection

    chromadb_manager_fixt._test_mock_collection.query.assert_called_once_with(
        query_texts=query_texts, n_results=n_results, include=['documents', 'metadatas', 'distances']
    )
    assert len(results) == 1
    assert results[0]['id'] == 'id1'
    assert results[0]['document'] == 'doc1 content'
    assert results[0]['metadata'] == {'source': 'test_source'}
    assert results[0]['distance'] == 0.01

@pytest.mark.asyncio
async def test_query_collection_async_no_results_found(chromadb_manager_fixt):
    chromadb_manager_fixt._test_mock_collection.query.return_value = {
        'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]
    }
    results = await chromadb_manager_fixt.query_collection(["query"], 1) # Renamed to query_collection
    assert results == []

@pytest.mark.asyncio
async def test_query_collection_async_malformed_response(chromadb_manager_fixt, caplog):
    caplog.set_level(logging.WARNING)
    # Simulate results where lists have different lengths
    chromadb_manager_fixt._test_mock_collection.query.return_value = {
        'ids': [['id1', 'id2']], # 2 ids
        'documents': [['doc1']], # 1 document
        'metadatas': [[{'s': 's1'}]], 'distances': [[0.1]]
    }
    results = await chromadb_manager_fixt.query_collection(["q"], 2) # Renamed to query_collection
    assert results == [] # Should return empty if data is inconsistent
    # Check for the more specific log message based on the new logic
    assert "Query results for query 'q' had mismatched lengths. IDs: 2, Docs: 1, Metadatas: 1, Distances: 1. Skipping result processing." in caplog.text

    # Simulate results missing a key
    caplog.clear() # Clear previous logs
    chromadb_manager_fixt._test_mock_collection.query.return_value = {'ids': [['id1']]} # Missing documents, metadatas, distances
    results = await chromadb_manager_fixt.query_collection(["q"], 1) # Renamed to query_collection
    assert results == []
    assert "Query results missing or malformed for key: documents" in caplog.text # Adjusted to check for 'documents' key


# --- Test get_collection_count_async ---
@pytest.mark.asyncio
async def test_get_collection_count_async(chromadb_manager_fixt):
    chromadb_manager_fixt._test_mock_collection.count.return_value = 42
    count = await chromadb_manager_fixt.get_collection_count_async()
    assert count == 42
    # __init__ calls count once, then get_collection_count_async calls it again.
    # We are interested in the call made by the method under test.
    chromadb_manager_fixt._test_mock_collection.count.assert_called_with() # Checks the last call

# --- Test clear_collection_async ---
@pytest.mark.asyncio
async def test_clear_collection_async(chromadb_manager_fixt, isolated_settings, caplog):
    caplog.set_level(logging.INFO)
    original_collection_name = chromadb_manager_fixt.collection_name

    # The client is chromadb_manager_fixt._test_mock_persistent_client
    chromadb_manager_fixt._test_mock_persistent_client.delete_collection.reset_mock()
    chromadb_manager_fixt._test_mock_persistent_client.get_or_create_collection.reset_mock() # Reset for this test

    # Mock for the client's get_or_create_collection for when it's called after delete
    new_mock_collection_after_clear = mock.MagicMock(name="NewMockCollectionAfterClear")
    chromadb_manager_fixt._test_mock_persistent_client.get_or_create_collection.return_value = new_mock_collection_after_clear

    await chromadb_manager_fixt.clear_collection_async()

    chromadb_manager_fixt._test_mock_persistent_client.delete_collection.assert_called_once_with(name=original_collection_name) # Expect keyword arg

    # get_or_create_collection is called by the SUT after delete_collection
    chromadb_manager_fixt._test_mock_persistent_client.get_or_create_collection.assert_called_once()
    call_args = chromadb_manager_fixt._test_mock_persistent_client.get_or_create_collection.call_args
    assert call_args[1]['name'] == original_collection_name
    assert isinstance(call_args[1]['embedding_function'], GeminiChromaEmbeddingFunction)

    assert chromadb_manager_fixt.collection is new_mock_collection_after_clear
    assert f"Collection '{original_collection_name}' cleared and recreated." in caplog.text

# --- Test _generate_id_from_text_and_source ---
def test_generate_id_from_text_and_source_consistency(chromadb_manager_fixt):
    text = "Hello, world!"
    source = "test_source.txt"
    expected_string_to_hash = f"{text}-{source}"
    # Implementation uses md5 and[:16]
    expected_id = "60686da7f523c6c4" # Correct MD5 hash prefix

    assert chromadb_manager_fixt.generate_id_from_text_and_source(text, source) == expected_id

def test_generate_id_from_text_and_source_uniqueness(chromadb_manager_fixt):
    id1 = chromadb_manager_fixt.generate_id_from_text_and_source("text1", "source1")
    id2 = chromadb_manager_fixt.generate_id_from_text_and_source("text2", "source1") # Different text
    id3 = chromadb_manager_fixt.generate_id_from_text_and_source("text1", "source2") # Different source
    id4 = chromadb_manager_fixt.generate_id_from_text_and_source("text1", "source1") # Same as id1

    assert id1 != id2
    assert id1 != id3
    assert id2 != id3
    assert id1 == id4
