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
    # Preserve the original settings object from the config module at the start of the test session
    # This is tricky because AppSettings() itself loads from env.
    # The goal is that llm.gemini_client.settings uses a version of settings
    # that reflects the env vars set for THAT specific test function.

    # Create a new AppSettings instance. It will load from the current state of os.environ.
    current_test_app_settings = AppSettings()

    # Monkeypatch the 'settings' object in the module where GeminiClient is defined
    monkeypatch.setattr("llm.gemini_client.settings", current_test_app_settings)

    # Also patch the 'settings' object in the config module itself, in case any other
    # part of the system indirectly uses it and expects it to be consistent.
    monkeypatch.setattr("config.settings", current_test_app_settings)

    yield current_test_app_settings


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mocks environment variables for GeminiClient settings for a typical success case."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test_api_key_from_env")
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "env-embedding-model")
    monkeypatch.setenv("GEMINI_GENERATION_MODEL", "env-generation-model")
    # The isolated_settings fixture, when called after this, will ensure AppSettings reloads with these.

@pytest.fixture
def gemini_client_fixt(mock_env_vars, isolated_settings, mocker): # Added mocker
    """Fixture to create a GeminiClient instance with mocked settings and genai."""

    # Mock attributes directly on the 'genai' object that llm.gemini_client has imported
    m_configure = mocker.patch.object(gemini_client_module.genai, 'configure', name="configure_mock")
    m_GenerativeModel_class = mocker.patch.object(gemini_client_module.genai, 'GenerativeModel', name="GenerativeModel_class_mock")

    # This is the mock for the *instance* of GenerativeModel
    m_generative_model_instance = mocker.MagicMock(name="GenerativeModel_instance_mock")
    m_GenerativeModel_class.return_value = m_generative_model_instance

    # Mock methods on the *instance*
    m_generative_model_instance.generate_content_async = mocker.AsyncMock(name="generate_content_async_mock")

    # Mock genai.embed_content as it's used by functools.partial in the client
    m_embed_content = mocker.patch.object(gemini_client_module.genai, 'embed_content', name="embed_content_mock")

    # `isolated_settings` has already run and patched `llm.gemini_client.settings`
    client = GeminiClient() # This will now use the patched genai.configure, genai.GenerativeModel etc.

    # Store mocks in the client instance for easy access in tests, if needed.
    client._test_mocks = {
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

def test_gemini_client_init_missing_api_key(monkeypatch, caplog): # No isolated_settings needed if AppSettings itself fails
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(Exception) as excinfo:
        AppSettings() # This will fail due to Pydantic validation in ApiKeysSettings
    assert "GOOGLE_API_KEY" in str(excinfo.value).upper()
    assert "field required" in str(excinfo.value).lower() or "must be set" in str(excinfo.value).lower()

# --- Tests for GeminiClient.generate_text_async ---

@pytest.mark.asyncio
async def test_generate_text_async_success(gemini_client_fixt, mocker): # Corrected indentation
    prompt = "Test prompt"
    expected_response_text = "Generated text response."
    # generate_content_async is on the model instance mock
    # Configure the mock to return an object that has a .text attribute with the desired string value
    mock_response_with_text = mocker.MagicMock()
    mock_response_with_text.text = expected_response_text
    gemini_client_fixt._test_mocks["model_instance"].generate_content_async.return_value = mock_response_with_text

    response = await gemini_client_fixt.generate_text_async(prompt)
    assert response == expected_response_text
    gemini_client_fixt._test_mocks["model_instance"].generate_content_async.assert_called_once_with(prompt)

@pytest.mark.asyncio
async def test_generate_text_async_api_error(gemini_client_fixt, caplog):
    prompt = "Error prompt"
    caplog.set_level(logging.ERROR)
    gemini_client_fixt._test_mocks["model_instance"].generate_content_async.side_effect = Exception("Gemini API Error")
    response = await gemini_client_fixt.generate_text_async(prompt)
    assert response == "Error: An unexpected error occurred during text generation. Gemini API Error"
    assert "Error generating text with Gemini: Gemini API Error" in caplog.text

@pytest.mark.asyncio
async def test_generate_text_async_empty_prompt(gemini_client_fixt, caplog):
    caplog.set_level(logging.WARNING)
    response = await gemini_client_fixt.generate_text_async("")
    # Client code for empty prompt in generate_text_async:
    # if not prompt or not prompt.strip():
    #   logger.warning("Empty prompt received. Returning empty response.")
    #   return "Error: Content generation failed for an unknown reason." (based on default path if no .text)
    # This part of client code was updated. It now returns "Error: Prompt cannot be empty."
    # For generate_text_async, if prompt is empty, it seems to fall through to a part where it tries to get .text
    # from a response that doesn't exist or is not formed correctly.
    # Let's assume it should hit the "No parts in response"
    assert response == "Error: Content generation failed for an unknown reason."
    assert "Empty prompt received. Returning empty response." in caplog.text # Check for the specific log
    gemini_client_fixt._test_mocks["model_instance"].generate_content_async.assert_not_called()

# --- Tests for GeminiClient.get_embedding_async ---

@pytest.mark.asyncio
async def test_get_embedding_async_success(gemini_client_fixt, isolated_settings):
    text_to_embed = "Embed this text."
    expected_embedding = [0.1, 0.2, 0.3]
    current_settings = isolated_settings

    # Configure the mock for genai.embed_content (which is client._test_mocks["embed_content"])
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
    assert embedding is None # Error case in client returns None
    assert f"Error generating embedding for text: '{text_to_embed[:50]}...': Gemini Embedding Error" in caplog.text

@pytest.mark.asyncio
async def test_get_embedding_async_empty_text(gemini_client_fixt, caplog):
    caplog.set_level(logging.WARNING)
    embedding = await gemini_client_fixt.get_embedding_async("")
    assert embedding is None # Empty text case in client returns None
    assert "Empty text provided for embedding, returning None." in caplog.text # Check client's log
    gemini_client_fixt._test_mocks["embed_content"].assert_not_called()

# --- Placeholder for Batch Embedding Tests ---
# (If GeminiClient gets a get_embedding_batch_async method)

# --- Sanity check for mock states after a test using the client fixture ---
def test_mock_state_after_fixture_use(gemini_client_fixt, isolated_settings):
    # This test runs after gemini_client_fixt has initialized a client.
    # It verifies that the __init__ calls happened on the mocks.
    current_settings = isolated_settings
    gemini_client_fixt._test_mocks["configure"].assert_called_once_with(
        api_key=current_settings.api_keys.google_api_key.get_secret_value()
    )
    gemini_client_fixt._test_mocks["GenerativeModel_class"].assert_called_once_with(
        current_settings.gemini_model.gemini_generation_model
    )
    # These should not have been called yet if the test itself didn't call client methods
    gemini_client_fixt._test_mocks["model_instance"].generate_content_async.assert_not_called()
    gemini_client_fixt._test_mocks["embed_content"].assert_not_called()
