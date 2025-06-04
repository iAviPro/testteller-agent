import pytest
import asyncio
from unittest import mock
import sys # For sys.modules mocking
import logging

# Import the module and client to be tested first
from llm.gemini_client import GeminiClient
import llm.gemini_client as gemini_client_module # To access the 'genai' alias used in the module
from config import settings as global_settings_config_module, AppSettings, ApiKeysSettings, GeminiModelSettings

# --- Fixture to manage global settings for tests ---
@pytest.fixture(scope="function") # No autouse, explicitly request it
def isolated_settings(monkeypatch):
    """
    Ensures each test function gets a fresh AppSettings instance, loaded from an environment
    optionally modified by other fixtures. Patches this instance into llm.gemini_client.settings.
    """
    current_test_app_settings = AppSettings()
    monkeypatch.setattr("llm.gemini_client.settings", current_test_app_settings)
    monkeypatch.setattr("config.settings", current_test_app_settings) # Also patch global if needed by other modules
    yield current_test_app_settings


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mocks environment variables for GeminiClient settings for a typical success case."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test_api_key_from_env")
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "env-embedding-model")
    monkeypatch.setenv("GEMINI_GENERATION_MODEL", "env-generation-model")

@pytest.fixture
def gemini_client_fixt(mock_env_vars, isolated_settings, mocker):
    """Fixture to create a GeminiClient instance with mocked settings and genai."""
    m_configure = mocker.patch.object(gemini_client_module.genai, 'configure', name="configure_mock")
    m_GenerativeModel_class = mocker.patch.object(gemini_client_module.genai, 'GenerativeModel', name="GenerativeModel_class_mock")
    m_generative_model_instance = mocker.MagicMock(name="GenerativeModel_instance_mock")
    m_GenerativeModel_class.return_value = m_generative_model_instance
    m_generative_model_instance.generate_content_async = mocker.AsyncMock(name="generate_content_async_mock")
    # Mock the synchronous generate_content method as well, as it's used by the client's generate_text_async
    m_generative_model_instance.generate_content = mocker.MagicMock(name="generate_content_sync_mock")
    m_embed_content = mocker.patch.object(gemini_client_module.genai, 'embed_content', name="embed_content_mock")

    client = GeminiClient()
    client._test_mocks = { # Store mocks for easy access in tests
        "configure": m_configure,
        "GenerativeModel_class": m_GenerativeModel_class,
        "model_instance": m_generative_model_instance,
        "embed_content": m_embed_content
    }
    return client

# --- Tests for GeminiClient.__init__ ---

def test_gemini_client_init_success(gemini_client_fixt, isolated_settings):
    current_settings = isolated_settings
    gemini_client_fixt._test_mocks["configure"].assert_called_once_with(
        api_key=current_settings.api_keys.google_api_key.get_secret_value()
    )
    gemini_client_fixt._test_mocks["GenerativeModel_class"].assert_called_once_with(
        current_settings.gemini_model.gemini_generation_model
    )
    assert gemini_client_fixt.generation_model is gemini_client_fixt._test_mocks["model_instance"]
    assert gemini_client_fixt.embedding_model_name == current_settings.gemini_model.gemini_embedding_model

def test_gemini_client_init_empty_api_key_from_settings(mock_env_vars, monkeypatch, caplog):
    # This test doesn't need isolated_settings directly if we are testing AppSettings loading
    # or if we patch the settings object used by GeminiClient constructor.
    # The goal is to make settings.api_keys.google_api_key.get_secret_value() return ""

    # Patch the specific setting access within the llm.gemini_client module
    # This requires that AppSettings successfully initializes first, then we modify its behavior.
    # To do this, we ensure GOOGLE_API_KEY is set to something valid for AppSettings init,
    # then specifically mock the get_secret_value() call on the *instance* of settings
    # that will be used by GeminiClient.

    # 1. Let AppSettings initialize normally (it will use "test_api_key_from_env" via mock_env_vars)
    #    The global `settings` object in config.py is what gets imported by llm.gemini_client
    #    We need to ensure this global `settings` object returns an empty key.

    # Create a fresh AppSettings that llm.gemini_client.settings will point to
    fresh_app_settings = AppSettings()
    monkeypatch.setattr(gemini_client_module, 'settings', fresh_app_settings)

    # Now, on these fresh_app_settings, mock get_secret_value
    # This is a bit convoluted because SecretStr is tricky.
    # Easier to mock the return of get_secret_value on the SecretStr instance.
    # Assuming api_keys.google_api_key is a SecretStr object.
    mock_secret_str_instance = fresh_app_settings.api_keys.google_api_key
    monkeypatch.setattr(mock_secret_str_instance, 'get_secret_value', mock.MagicMock(return_value=""))

    caplog.set_level(logging.ERROR)
    with pytest.raises(ValueError, match="Google API Key is not set or is empty"):
        GeminiClient()

    assert any("Google API Key is not set or is empty" in record.message and record.levelname == "ERROR" for record in caplog.records), \
           "Log message for empty API key not found."

# --- Tests for GeminiClient.generate_text_async ---

@pytest.mark.asyncio
async def test_generate_text_async_success(gemini_client_fixt, mocker):
    prompt = "Test prompt"
    expected_response_text = "Generated text response."
    mock_response_with_text = mocker.MagicMock()
    mock_response_with_text.text = expected_response_text
    # Configure the synchronous generate_content method on the model_instance mock
    gemini_client_fixt._test_mocks["model_instance"].generate_content.return_value = mock_response_with_text

    response = await gemini_client_fixt.generate_text_async(prompt)
    assert response == expected_response_text
    gemini_client_fixt._test_mocks["model_instance"].generate_content.assert_called_once_with(
        prompt,
        generation_config=mocker.ANY,
        safety_settings=mocker.ANY
    )

@pytest.mark.asyncio
async def test_generate_text_async_api_error(gemini_client_fixt, caplog):
    prompt = "Error prompt"
    caplog.set_level(logging.ERROR)
    gemini_client_fixt._test_mocks["model_instance"].generate_content.side_effect = Exception("Gemini API Error")
    response = await gemini_client_fixt.generate_text_async(prompt)
    assert response == "Error: An unexpected error occurred during text generation. Gemini API Error"
    assert "Error generating text: Gemini API Error" in caplog.text # Corrected log message check

@pytest.mark.asyncio
async def test_generate_text_async_empty_prompt(gemini_client_fixt, mocker, caplog):
    caplog.set_level(logging.ERROR) # Client logs this as ERROR

    model_instance_mock = gemini_client_fixt._test_mocks["model_instance"]

    # Mock the response from generate_content when prompt is empty
    mock_llm_response = mocker.MagicMock(name="MockLLMResponseForEmptyPrompt")
    mock_llm_response.parts = []  # No parts
    mock_llm_response.prompt_feedback = None # No specific block reason
    # The .text attribute would normally be derived from parts. If parts is empty, this might not be accessed directly
    # by client code if it first checks 'parts'. If client does access .text, then:
    # type(mock_llm_response).text = mocker.PropertyMock(return_value="")

    model_instance_mock.generate_content.return_value = mock_llm_response

    response = await gemini_client_fixt.generate_text_async("")

    assert response == "Error: Content generation failed for an unknown reason."

    model_instance_mock.generate_content.assert_called_once_with(
        "",  # Empty prompt
        generation_config=mocker.ANY,
        safety_settings=mocker.ANY
    )
    # Check for the specific log message from the client
    assert "Content generation failed: No parts in response and no block reason provided." in caplog.text

@pytest.mark.asyncio
async def test_generate_text_async_content_blocked(gemini_client_fixt, mocker, caplog):
    prompt = "A potentially problematic prompt"
    caplog.set_level(logging.ERROR)

    mock_sync_generate_content = gemini_client_fixt._test_mocks["model_instance"].generate_content

    # Simulate a response that is blocked
    mock_llm_response_blocked = mocker.MagicMock(name="MockLLMResponseBlocked")
    mock_llm_response_blocked.parts = []  # Important: no parts for blocked content

    # Setup prompt_feedback attribute
    mock_feedback = mocker.MagicMock(name="MockPromptFeedback")
    mock_feedback.block_reason = "SAFETY"
    mock_feedback.block_reason_message = "Blocked due to safety reasons." # More specific message
    mock_llm_response_blocked.prompt_feedback = mock_feedback

    # If client tries to access .text on a blocked response (it shouldn't if parts is checked first)
    # type(mock_llm_response_blocked).text = mocker.PropertyMock(return_value="")

    mock_sync_generate_content.return_value = mock_llm_response_blocked

    response_text = await gemini_client_fixt.generate_text_async(prompt)

    expected_error_message_client = "Error: Content generation blocked. Reason: Blocked due to safety reasons."
    assert response_text == expected_error_message_client

    assert "Content generation blocked. Reason: Blocked due to safety reasons." in caplog.text

    mock_sync_generate_content.assert_called_once_with(
        prompt,
        generation_config=mocker.ANY,
        safety_settings=mocker.ANY
    )

# --- Tests for GeminiClient.get_embedding_async ---

@pytest.mark.asyncio
async def test_get_embedding_async_success(gemini_client_fixt, isolated_settings):
    text_to_embed = "Embed this text."
    expected_embedding = [0.1, 0.2, 0.3]
    current_settings = isolated_settings
    gemini_client_fixt._test_mocks["embed_content"].return_value = {'embedding': expected_embedding}

    embedding = await gemini_client_fixt.get_embedding_async(text_to_embed)

    gemini_client_fixt._test_mocks["embed_content"].assert_called_once_with(
        model=current_settings.gemini_model.gemini_embedding_model,
        content=text_to_embed,
        task_type="RETRIEVAL_DOCUMENT"
    )
    assert embedding == expected_embedding

@pytest.mark.asyncio
async def test_get_embedding_async_api_error(gemini_client_fixt, caplog):
    text_to_embed = "Error embedding this."
    caplog.set_level(logging.ERROR)
    gemini_client_fixt._test_mocks["embed_content"].side_effect = Exception("Gemini Embedding Error")
    embedding = await gemini_client_fixt.get_embedding_async(text_to_embed)
    assert embedding is None
    assert f"Error generating embedding for text: '{text_to_embed[:50]}...': Gemini Embedding Error" in caplog.text

@pytest.mark.asyncio
async def test_get_embedding_async_empty_text(gemini_client_fixt, caplog):
    caplog.set_level(logging.WARNING)
    embedding = await gemini_client_fixt.get_embedding_async("")
    assert embedding is None
    assert "Empty text provided for embedding, returning None." in caplog.text
    gemini_client_fixt._test_mocks["embed_content"].assert_not_called()

# --- Tests for GeminiClient.get_embedding_sync ---

def test_get_embedding_sync_success(gemini_client_fixt, isolated_settings, mocker): # Added mocker for consistency if needed
    text_to_embed = "Embed this text synchronously."
    expected_embedding = [0.4, 0.5, 0.6]
    current_settings = isolated_settings

    gemini_client_fixt._test_mocks["embed_content"].reset_mock()
    gemini_client_fixt._test_mocks["embed_content"].side_effect = None
    gemini_client_fixt._test_mocks["embed_content"].return_value = {'embedding': expected_embedding}

    embedding = gemini_client_fixt.get_embedding_sync(text_to_embed)

    gemini_client_fixt._test_mocks["embed_content"].assert_called_once_with(
        model=current_settings.gemini_model.gemini_embedding_model,
        content=text_to_embed,
        task_type="RETRIEVAL_DOCUMENT"
    )
    assert embedding == expected_embedding

def test_get_embedding_sync_api_error(gemini_client_fixt, caplog):
    text_to_embed = "Error embedding this synchronously."
    caplog.set_level(logging.ERROR)

    gemini_client_fixt._test_mocks["embed_content"].reset_mock()
    gemini_client_fixt._test_mocks["embed_content"].side_effect = Exception("Gemini Sync Embedding Error")

    embedding = gemini_client_fixt.get_embedding_sync(text_to_embed)

    assert embedding is None
    assert f"Error generating sync embedding for text: '{text_to_embed[:50]}...': Gemini Sync Embedding Error" in caplog.text
    gemini_client_fixt._test_mocks["embed_content"].assert_called_once()

def test_get_embedding_sync_empty_text(gemini_client_fixt, caplog):
    caplog.set_level(logging.WARNING)
    gemini_client_fixt._test_mocks["embed_content"].reset_mock()
    # Clear any side effect that might have been set by other tests
    gemini_client_fixt._test_mocks["embed_content"].side_effect = None # Ensure no leftover side effects

    embedding = gemini_client_fixt.get_embedding_sync("")

    assert embedding is None
    assert "Empty text provided for sync embedding, returning None." in caplog.text # Specific log
    gemini_client_fixt._test_mocks["embed_content"].assert_not_called() # Should not be called

# --- Placeholder for Batch Embedding Tests ---

# --- Sanity check for mock states after a test using the client fixture ---
def test_mock_state_after_fixture_use(gemini_client_fixt, isolated_settings):
    current_settings = isolated_settings
    gemini_client_fixt._test_mocks["configure"].assert_called_once_with(
        api_key=current_settings.api_keys.google_api_key.get_secret_value()
    )
    gemini_client_fixt._test_mocks["GenerativeModel_class"].assert_called_once_with(
        current_settings.gemini_model.gemini_generation_model
    )
    gemini_client_fixt._test_mocks["model_instance"].generate_content_async.assert_not_called()
    gemini_client_fixt._test_mocks["model_instance"].generate_content.assert_not_called()
    gemini_client_fixt._test_mocks["embed_content"].assert_not_called()


# --- Tests for get_embeddings_async ---

@pytest.mark.asyncio
async def test_get_embeddings_async_success(gemini_client_fixt, mocker):
    texts = ["text1", "text2", "text3"]
    expected_embeddings = [[0.1], [0.2], [0.3]]

    # Mock the single embedding method that the batch method calls
    gemini_client_fixt.get_embedding_async = mocker.AsyncMock(
        side_effect=lambda text: expected_embeddings[texts.index(text)] if text in texts else None
    )

    results = await gemini_client_fixt.get_embeddings_async(texts)

    assert results == expected_embeddings
    assert gemini_client_fixt.get_embedding_async.call_count == len(texts)
    for text in texts:
        gemini_client_fixt.get_embedding_async.assert_any_call(text)

@pytest.mark.asyncio
async def test_get_embeddings_async_partial_failure(gemini_client_fixt, mocker, caplog):
    caplog.set_level(logging.ERROR)
    texts = ["text1", "fail_text2", "text3"]
    # text1 maps to [0.1], fail_text2 will raise Exception, text3 maps to [0.3]
    # The side_effect needs to handle the calls based on input text.
    async def side_effect_func(text):
        if text == "text1":
            return [0.1]
        elif text == "fail_text2":
            raise Exception("Simulated embedding failure for fail_text2")
        elif text == "text3":
            return [0.3]
        return None

    gemini_client_fixt.get_embedding_async = mocker.AsyncMock(side_effect=side_effect_func)

    results = await gemini_client_fixt.get_embeddings_async(texts)

    assert results == [[0.1], None, [0.3]] # None for the failed one
    assert gemini_client_fixt.get_embedding_async.call_count == len(texts)
    assert "Failed to get embedding for text chunk 1 after retries: Simulated embedding failure for fail_text2" in caplog.text

@pytest.mark.asyncio
async def test_get_embeddings_async_empty_list(gemini_client_fixt, mocker):
    gemini_client_fixt.get_embedding_async = mocker.AsyncMock() # So we can check call_count
    results = await gemini_client_fixt.get_embeddings_async([])
    assert results == []
    gemini_client_fixt.get_embedding_async.assert_not_called()

@pytest.mark.asyncio
async def test_get_embeddings_async_all_fail(gemini_client_fixt, mocker, caplog):
    caplog.set_level(logging.ERROR)
    texts = ["fail1", "fail2"]

    gemini_client_fixt.get_embedding_async = mocker.AsyncMock(
        side_effect=Exception("Simulated general embedding failure")
    )

    results = await gemini_client_fixt.get_embeddings_async(texts)

    assert results == [None, None]
    assert gemini_client_fixt.get_embedding_async.call_count == len(texts)
    assert "Failed to get embedding for text chunk 0 after retries: Simulated general embedding failure" in caplog.text
    assert "Failed to get embedding for text chunk 1 after retries: Simulated general embedding failure" in caplog.text

# --- Tests for get_embeddings_sync ---

def test_get_embeddings_sync_success(gemini_client_fixt, mocker):
    texts = ["text1_sync", "text2_sync"]
    expected_embeddings = [[0.11], [0.22]]

    gemini_client_fixt.get_embedding_sync = mocker.MagicMock(
        side_effect=lambda text: expected_embeddings[texts.index(text)] if text in texts else None
    )

    results = gemini_client_fixt.get_embeddings_sync(texts)

    assert results == expected_embeddings
    assert gemini_client_fixt.get_embedding_sync.call_count == len(texts)
    for text in texts:
        gemini_client_fixt.get_embedding_sync.assert_any_call(text)

def test_get_embeddings_sync_partial_failure(gemini_client_fixt, mocker, caplog):
    caplog.set_level(logging.ERROR) # Assuming get_embedding_sync logs errors if underlying call fails
    texts = ["text1_sync", "fail_text2_sync", "text3_sync"]

    def side_effect_func(text):
        if text == "text1_sync":
            return [0.11]
        elif text == "fail_text2_sync":
            # Simulate an error that get_embedding_sync itself might log and then return None
            # For this test, we assume get_embedding_sync handles the error and returns None
            # The logging of the error would happen inside get_embedding_sync
            # We are testing the batch aggregation here.
            return None
        elif text == "text3_sync":
            return [0.33]
        return None

    gemini_client_fixt.get_embedding_sync = mocker.MagicMock(side_effect=side_effect_func)

    results = gemini_client_fixt.get_embeddings_sync(texts)

    assert results == [[0.11], None, [0.33]]
    assert gemini_client_fixt.get_embedding_sync.call_count == len(texts)
    # If get_embedding_sync logs its own errors, those would be checked here.
    # The batch method get_embeddings_sync itself doesn't add more logging for individual failures.

def test_get_embeddings_sync_empty_list(gemini_client_fixt, mocker):
    gemini_client_fixt.get_embedding_sync = mocker.MagicMock()
    results = gemini_client_fixt.get_embeddings_sync([])
    assert results == []
    gemini_client_fixt.get_embedding_sync.assert_not_called()

def test_get_embeddings_sync_all_fail(gemini_client_fixt, mocker, caplog):
    caplog.set_level(logging.ERROR) # Assuming get_embedding_sync logs errors
    texts = ["fail1_sync", "fail2_sync"]

    gemini_client_fixt.get_embedding_sync = mocker.MagicMock(return_value=None) # All individual calls return None

    results = gemini_client_fixt.get_embeddings_sync(texts)

    assert results == [None, None]
    assert gemini_client_fixt.get_embedding_sync.call_count == len(texts)
    # Logging would be inside get_embedding_sync if individual calls fail/return None due to error.
    # The batch method itself doesn't add extra logs for all failing.
