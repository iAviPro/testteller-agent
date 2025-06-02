import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config import get_settings # Changed to get_settings

logger = logging.getLogger(__name__)

import httpx # Ensure httpx is imported

# Define common transient error types for APIs
# This list might need to be expanded based on specific API client exceptions
TRANSIENT_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    httpx.NetworkError,
    # Add specific exceptions from google.generativeai or other libraries if known
    # e.g., from google.api_core.exceptions import TooManyRequests, ServiceUnavailable
)
# print(f"DEBUG: TRANSIENT_EXCEPTIONS in retry_helpers: {TRANSIENT_EXCEPTIONS}")

def log_retry_attempt(retry_state):
    # Access settings inside the function to ensure it's fresh
    current_settings = get_settings()
    logger.warning(
        "Retrying %s due to %s, attempt %d of %d. Waiting %.2fs before next attempt.",
        retry_state.fn.__name__,
        retry_state.outcome.exception(), # Use .exception() for the actual exception instance
        retry_state.attempt_number,
        current_settings.api_retry.api_retry_attempts,
        retry_state.next_action.sleep
    )

# Helper functions to defer settings access until tenacity needs them
def _get_stop_config(retry_state=None): # Must accept retry_state
    # print(f"DEBUG: _get_stop_config called. Retrying on: {TRANSIENT_EXCEPTIONS}") # Debug
    return stop_after_attempt(get_settings().api_retry.api_retry_attempts)

def _get_wait_config(retry_state=None): # Must accept retry_state
    return wait_exponential(
        multiplier=get_settings().api_retry.api_retry_wait_seconds, min=1, max=10
    )

# Generic retry decorator for API calls
api_retry_async = retry(
    stop=_get_stop_config,
    wait=_get_wait_config,
    retry=retry_if_exception_type(TRANSIENT_EXCEPTIONS), # Standard predicate
    before_sleep=log_retry_attempt,
    reraise=True  # Reraise the exception if all retries fail
)

api_retry_sync = retry(
    stop=_get_stop_config,
    wait=_get_wait_config,
    retry=retry_if_exception_type(TRANSIENT_EXCEPTIONS), # Standard predicate
    before_sleep=log_retry_attempt,
    reraise=True
)
