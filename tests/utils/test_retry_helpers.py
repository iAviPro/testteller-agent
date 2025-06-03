import pytest
import logging
import asyncio
from unittest import mock

from tenacity import RetryError, stop_after_attempt, wait_exponential, retry_if_exception_type


# Import the module to be tested *after* potentially patching settings if needed,
# or ensure the module can be reloaded/decorators reconfigured.
# For this structure, we'll import it and then the fixture will patch settings
# and reconfigure the decorators directly.
import utils.retry_helpers
from utils.retry_helpers import log_retry_attempt, TRANSIENT_EXCEPTIONS

# Fixture to mock settings for retry tests
@pytest.fixture
def mock_retry_settings(monkeypatch, mocker): # Added mocker, though not used in current impl
    class MockApiRetrySettings:
        def __init__(self, attempts=3, wait_seconds=1):
            self.api_retry_attempts = attempts
            self.api_retry_wait_seconds = wait_seconds

    class MockSettings:
        def __init__(self, retry_attempts=3, retry_wait_seconds=1):
            self.api_retry = MockApiRetrySettings(attempts=retry_attempts, wait_seconds=retry_wait_seconds)

    def _mock_settings(retry_attempts=3, retry_wait_seconds=0.01): # Use small wait for tests
        settings_instance = MockSettings(retry_attempts, retry_wait_seconds)

        # Patch the 'settings' object within the 'utils.retry_helpers' module
        monkeypatch.setattr("utils.retry_helpers.settings", settings_instance)

        # Re-configure the decorators by creating new instances or updating their parameters.
        # Tenacity decorators are typically functions that return a decorator object.
        # The original decorators in retry_helpers are module-level variables.
        # We need to replace these variables with newly configured decorator instances.

        # Re-create/re-assign sync decorator
        # Assuming 'retry' from tenacity is available as utils.retry_helpers.retry
        utils.retry_helpers.api_retry_sync = utils.retry_helpers.retry(
            stop=stop_after_attempt(settings_instance.api_retry.api_retry_attempts),
            wait=wait_exponential(
                multiplier=settings_instance.api_retry.api_retry_wait_seconds, min=0.01, max=0.1), # min/max for speed
            retry=retry_if_exception_type(TRANSIENT_EXCEPTIONS),
            before_sleep=log_retry_attempt,
            reraise=True
        )

        # Re-create/re-assign async decorator
        utils.retry_helpers.api_retry_async = utils.retry_helpers.retry(
            stop=stop_after_attempt(settings_instance.api_retry.api_retry_attempts),
            wait=wait_exponential(
                multiplier=settings_instance.api_retry.api_retry_wait_seconds, min=0.01, max=0.1),
            retry=retry_if_exception_type(TRANSIENT_EXCEPTIONS),
            before_sleep=log_retry_attempt,
            reraise=True
        )
        return settings_instance

    return _mock_settings


# --- Test log_retry_attempt ---
def test_log_retry_attempt(caplog, mock_retry_settings): # Add mock_retry_settings to ensure settings are patched
    # Set specific attempts for this test via settings
    current_settings = mock_retry_settings(retry_attempts=5)

    class MockException(Exception):
        pass

    mock_fn = mock.MagicMock(__name__="test_function")
    mock_outcome = mock.MagicMock()
    mock_outcome.exception.return_value = MockException("Test error")

    mock_retry_state = mock.MagicMock()
    mock_retry_state.fn = mock_fn
    mock_retry_state.outcome = mock_outcome
    mock_retry_state.attempt_number = 2
    mock_retry_state.next_action.sleep = 0.5 # seconds

    with caplog.at_level(logging.WARNING):
        log_retry_attempt(mock_retry_state)

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == "WARNING"
    assert "Retrying test_function" in record.message
    assert "due to Test error" in record.message
    assert f"attempt 2 of {current_settings.api_retry.api_retry_attempts}" in record.message # Check against retry_attempts from settings
    assert "Waiting 0.50s before next attempt" in record.message


# --- Helper for retry tests ---
class NonTransientError(Exception):
    pass

class TransientError(ConnectionError): # Inherits from one of TRANSIENT_EXCEPTIONS
    pass


# --- Tests for api_retry_sync ---

def test_api_retry_sync_success_first_try(mock_retry_settings):
    mock_retry_settings() # Apply default mock settings
    mock_func = mock.MagicMock(return_value="success")

    @utils.retry_helpers.api_retry_sync # Use the (potentially reconfigured) decorator
    def decorated_func():
        return mock_func()

    assert decorated_func() == "success"
    mock_func.assert_called_once()


def test_api_retry_sync_success_after_retries(mock_retry_settings, caplog):
    num_attempts = 3
    mock_retry_settings(retry_attempts=num_attempts, retry_wait_seconds=0.01)
    mock_func = mock.MagicMock()
    mock_func.side_effect = [TransientError("fail1"), TransientError("fail2"), "success"]

    @utils.retry_helpers.api_retry_sync
    def decorated_func():
        return mock_func()

    with caplog.at_level(logging.WARNING):
        assert decorated_func() == "success"

    assert mock_func.call_count == num_attempts
    assert len(caplog.records) == num_attempts - 1


def test_api_retry_sync_failure_all_retries(mock_retry_settings, caplog):
    num_attempts = 2
    mock_retry_settings(retry_attempts=num_attempts, retry_wait_seconds=0.01)
    mock_func = mock.MagicMock(side_effect=TransientError("persistent failure"))

    @utils.retry_helpers.api_retry_sync
    def decorated_func():
        return mock_func()

    # If reraise=True, the original exception should be raised after retries are exhausted.
    with pytest.raises(TransientError) as excinfo:
        with caplog.at_level(logging.WARNING):
            decorated_func()

    assert "persistent failure" in str(excinfo.value) # Check the content of the raised TransientError
    assert mock_func.call_count == num_attempts
    assert len(caplog.records) == num_attempts - 1 # log_retry_attempt is called before sleep


def test_api_retry_sync_non_transient_error_no_retry(mock_retry_settings):
    mock_retry_settings()
    mock_func = mock.MagicMock(side_effect=NonTransientError("don't retry this"))

    @utils.retry_helpers.api_retry_sync
    def decorated_func():
        return mock_func()

    with pytest.raises(NonTransientError):
        decorated_func()
    mock_func.assert_called_once()


# --- Tests for api_retry_async ---

@pytest.mark.asyncio
async def test_api_retry_async_success_first_try(mock_retry_settings):
    mock_retry_settings()
    mock_async_func = mock.AsyncMock(return_value="success_async")

    @utils.retry_helpers.api_retry_async
    async def decorated_async_func():
        return await mock_async_func()

    assert await decorated_async_func() == "success_async"
    mock_async_func.assert_called_once()


@pytest.mark.asyncio
async def test_api_retry_async_success_after_retries(mock_retry_settings, caplog):
    num_attempts = 3
    mock_retry_settings(retry_attempts=num_attempts, retry_wait_seconds=0.01)
    mock_async_func = mock.AsyncMock()
    mock_async_func.side_effect = [TransientError("async_fail1"), TransientError("async_fail2"), "success_async"]

    @utils.retry_helpers.api_retry_async
    async def decorated_async_func():
        return await mock_async_func()

    with caplog.at_level(logging.WARNING):
        assert await decorated_async_func() == "success_async"

    assert mock_async_func.call_count == num_attempts
    assert len(caplog.records) == num_attempts - 1


@pytest.mark.asyncio
async def test_api_retry_async_failure_all_retries(mock_retry_settings, caplog):
    num_attempts = 2
    mock_retry_settings(retry_attempts=num_attempts, retry_wait_seconds=0.01)
    mock_async_func = mock.AsyncMock(side_effect=TransientError("async_persistent_failure"))

    @utils.retry_helpers.api_retry_async
    async def decorated_async_func():
        return await mock_async_func()

    # If reraise=True, the original exception should be raised after retries are exhausted.
    with pytest.raises(TransientError) as excinfo:
        with caplog.at_level(logging.WARNING):
            await decorated_async_func()

    assert "async_persistent_failure" in str(excinfo.value) # Check the content of the raised TransientError
    assert mock_async_func.call_count == num_attempts
    assert len(caplog.records) == num_attempts - 1 # log_retry_attempt is called before sleep


@pytest.mark.asyncio
async def test_api_retry_async_non_transient_error_no_retry(mock_retry_settings):
    mock_retry_settings()
    mock_async_func = mock.AsyncMock(side_effect=NonTransientError("async_dont_retry"))

    @utils.retry_helpers.api_retry_async
    async def decorated_async_func():
        return await mock_async_func()

    with pytest.raises(NonTransientError):
        await decorated_async_func()
    mock_async_func.assert_called_once()
