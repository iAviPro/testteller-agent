import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from testteller.config import settings

logger = logging.getLogger(__name__)

# Define common transient error types for APIs
# This list might need to be expanded based on specific API client exceptions
TRANSIENT_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    # Add specific exceptions from google.generativeai or other libraries if known
    # e.g., from google.api_core.exceptions import TooManyRequests, ServiceUnavailable
)


def log_retry_attempt(retry_state):
    logger.warning(
        "Retrying %s due to %s, attempt %d of %d. Waiting %.2fs before next attempt.",
        retry_state.fn.__name__,
        retry_state.outcome.exception(),
        retry_state.attempt_number,
        settings.api_retry_attempts,
        retry_state.next_action.sleep
    )


# Generic retry decorator for API calls
api_retry_async = retry(
    stop=stop_after_attempt(settings.api_retry.api_retry_attempts),
    wait=wait_exponential(
        multiplier=settings.api_retry.api_retry_wait_seconds, min=1, max=10
    ),
    retry=retry_if_exception_type(TRANSIENT_EXCEPTIONS),
    before_sleep=log_retry_attempt,
    reraise=True  # Reraise the exception if all retries fail
)

api_retry_sync = retry(
    stop=stop_after_attempt(settings.api_retry.api_retry_attempts),
    wait=wait_exponential(
        multiplier=settings.api_retry.api_retry_wait_seconds, min=1, max=10
    ),
    retry=retry_if_exception_type(TRANSIENT_EXCEPTIONS),
    before_sleep=log_retry_attempt,
    reraise=True
)
