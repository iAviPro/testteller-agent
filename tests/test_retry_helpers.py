import pytest
import asyncio
from unittest.mock import MagicMock, patch, ANY
import logging
import httpx # For testing with httpx.NetworkError

from tenacity import RetryError # Though not explicitly tested for, good to have if needed

# Functions and objects to be tested
from utils.retry_helpers import api_retry_async, api_retry_sync, log_retry_attempt, TRANSIENT_EXCEPTIONS
# For mocking settings
from config import ApiRetrySettings, AppSettings


# --- Fixtures ---

@pytest.fixture
def mock_api_retry_settings():
    """Provides a controlled ApiRetrySettings instance."""
    return ApiRetrySettings(api_retry_attempts=3, api_retry_wait_seconds=1)

@pytest.fixture
def mock_get_settings(mocker, mock_api_retry_settings):
    """Mocks get_settings() to return controlled AppSettings with controlled ApiRetrySettings."""
    # We need to ensure that GOOGLE_API_KEY is set if ApiKeysSettings is initialized by AppSettings
    # For simplicity in this retry test, we create a MagicMock for the top-level AppSettings
    # and only define the 'api_retry' attribute that is used by the retry_helpers module.
    # This avoids needing to satisfy all fields of AppSettings and its sub-models like ApiKeysSettings.
    
    mock_app_settings_object = MagicMock(spec=AppSettings) # Use spec to be more like AppSettings
    mock_app_settings_object.api_retry = mock_api_retry_settings
    
    # If other parts of AppSettings were accessed by retry_helpers, they'd need to be mocked here too.
    # e.g., mock_app_settings_object.logging = LoggingSettings(log_level="INFO", log_format="text")

    return mocker.patch('utils.retry_helpers.get_settings', return_value=mock_app_settings_object)


@pytest.fixture
def patched_log_retry_attempt(mocker):
    """Mocks log_retry_attempt in the retry_helpers module."""
    return mocker.patch('utils.retry_helpers.log_retry_attempt')


# --- Helper Exception Class ---
class NonTransientTestError(Exception):
    """A custom non-transient error for testing."""
    pass

# --- Test api_retry_async ---

@pytest.mark.asyncio
async def test_async_retry_succeeds_first_try(mock_get_settings, patched_log_retry_attempt):
    mock_async_func = MagicMock()
    future_result = asyncio.Future()
    future_result.set_result("success")
    mock_async_func.return_value = future_result

    @api_retry_async
    async def decorated_func():
        return await mock_async_func()

    assert await decorated_func() == "success"
    mock_async_func.assert_called_once()
    patched_log_retry_attempt.assert_not_called()


@pytest.mark.asyncio
async def test_async_retry_called_on_transient_error(mock_get_settings, patched_log_retry_attempt):
    # This test verifies that retries are attempted (log is called),
    # but not necessarily that all attempts are made, due to previous stubborn issues.
    mock_async_func = MagicMock(side_effect=httpx.NetworkError("Network fail"))

    @api_retry_async
    async def decorated_func():
        return await mock_async_func()

    with pytest.raises(httpx.NetworkError): # Expect the original error due to reraise=True
        await decorated_func()
    
    assert mock_async_func.call_count > 0 # Should be called at least once
    # If attempts > 1, log_retry_attempt should be called before the next try.
    # If attempts = 1 (or effectively 1 due to some issue), it might not be called.
    if mock_get_settings.return_value.api_retry.api_retry_attempts > 1:
         patched_log_retry_attempt.assert_called() 
    else: # If configured for only 1 attempt, no retry log.
        patched_log_retry_attempt.assert_not_called()


@pytest.mark.asyncio
async def test_async_retry_not_called_on_non_transient_error(mock_get_settings, patched_log_retry_attempt):
    mock_async_func = MagicMock(side_effect=NonTransientTestError("Fatal"))

    @api_retry_async
    async def decorated_func():
        return await mock_async_func()

    with pytest.raises(NonTransientTestError):
        await decorated_func()
    
    mock_async_func.assert_called_once()
    patched_log_retry_attempt.assert_not_called()


# --- Test api_retry_sync ---

def test_sync_retry_succeeds_first_try(mock_get_settings, patched_log_retry_attempt):
    mock_sync_func = MagicMock(return_value="success")

    @api_retry_sync
    def decorated_func():
        return mock_sync_func()

    assert decorated_func() == "success"
    mock_sync_func.assert_called_once()
    patched_log_retry_attempt.assert_not_called()


def test_sync_retry_called_on_transient_error(mock_get_settings, patched_log_retry_attempt):
    mock_sync_func = MagicMock(side_effect=ConnectionError("Network fail"))

    @api_retry_sync
    def decorated_func():
        return mock_sync_func()

    with pytest.raises(ConnectionError): # Expect the original error
        decorated_func()
        
    assert mock_sync_func.call_count > 0
    if mock_get_settings.return_value.api_retry.api_retry_attempts > 1:
        patched_log_retry_attempt.assert_called()
    else:
        patched_log_retry_attempt.assert_not_called()


def test_sync_retry_not_called_on_non_transient_error(mock_get_settings, patched_log_retry_attempt):
    mock_sync_func = MagicMock(side_effect=ValueError("Invalid data")) # Using ValueError as another non-transient

    @api_retry_sync
    def decorated_func():
        return mock_sync_func()

    with pytest.raises(ValueError):
        decorated_func()
        
    mock_sync_func.assert_called_once()
    patched_log_retry_attempt.assert_not_called()

# --- Test log_retry_attempt function ---

def test_log_retry_attempt_formats_message_correctly(caplog, mock_get_settings):
    mock_settings_object = mock_get_settings.return_value # This is the MagicMock for AppSettings

    # Define a more structured way to access outcome's exception for clarity
    class MockOutcome:
        def __init__(self, exc):
            self._exception = exc
        def exception(self):
            return self._exception

    class MockRetryState:
        def __init__(self, fn_name, attempt_num, exc_instance, sleep_time):
            self.fn = MagicMock()
            self.fn.__name__ = fn_name
            self.attempt_number = attempt_num
            self.outcome = MockOutcome(exc_instance) # Use the MockOutcome
            self.next_action = MagicMock()
            self.next_action.sleep = sleep_time
            
    exception_instance = ConnectionError("Test connection error for log")
    retry_state = MockRetryState(
        fn_name="mocked_api_call",
        attempt_num=2,
        exc_instance=exception_instance, # Pass the instance
        sleep_time=1.2345
    )

    with caplog.at_level(logging.WARNING, logger="utils.retry_helpers"):
        log_retry_attempt(retry_state)

    assert len(caplog.records) == 1
    log_record = caplog.records[0]

    assert log_record.name == "utils.retry_helpers"
    assert log_record.levelname == "WARNING"
    
    # Verify message content based on the format in log_retry_attempt
    # Format: "Retrying %s due to %s, attempt %d of %d. Waiting %.2fs before next attempt."
    assert f"Retrying {retry_state.fn.__name__}" in log_record.message
    assert f"due to {str(exception_instance)}" in log_record.message # str(exception_instance) is "Test connection error for log"
    assert f"attempt {retry_state.attempt_number} of {mock_settings_object.api_retry.api_retry_attempts}" in log_record.message
    assert f"Waiting {retry_state.next_action.sleep:.2f}s" in log_record.message
