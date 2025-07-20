"""
Retry decorators for API calls using tenacity.
"""
import asyncio
import logging
from functools import wraps
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryCallState,
    RetryError
)
from testteller.config import settings

logger = logging.getLogger(__name__)

# Default retry configuration
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_WAIT_SECONDS = 2
DEFAULT_RETRY_MAX_WAIT_SECONDS = 10  # Cap the exponential backoff


def get_retry_config():
    """Get retry configuration from settings or use defaults."""
    try:
        if settings and settings.api_retry:
            retry_attempts = settings.api_retry.__dict__.get(
                'api_retry_attempts', DEFAULT_RETRY_ATTEMPTS)
            retry_wait_seconds = settings.api_retry.__dict__.get(
                'api_retry_wait_seconds', DEFAULT_RETRY_WAIT_SECONDS)
        else:
            retry_attempts = DEFAULT_RETRY_ATTEMPTS
            retry_wait_seconds = DEFAULT_RETRY_WAIT_SECONDS
            logger.info("Using default retry configuration: attempts=%d, wait=%d",
                        retry_attempts, retry_wait_seconds)
    except Exception as e:
        logger.warning(
            "Failed to get retry configuration from settings: %s. Using defaults.", e)
        retry_attempts = DEFAULT_RETRY_ATTEMPTS
        retry_wait_seconds = DEFAULT_RETRY_WAIT_SECONDS

    return retry_attempts, retry_wait_seconds


def before_sleep(retry_state: RetryCallState):
    """Log retry attempts."""
    if retry_state.outcome is not None and retry_state.outcome.failed:
        exception = retry_state.outcome.exception()
        logger.warning(
            "Retrying after error: %s. Attempt %d/%d",
            str(exception),
            retry_state.attempt_number,
            get_retry_config()[0]  # Get max attempts
        )


# Retry decorator for synchronous functions
api_retry_sync = retry(
    retry=retry_if_exception_type((Exception,)),
    stop=stop_after_attempt(get_retry_config()[0]),
    wait=wait_exponential(
        multiplier=get_retry_config()[1],
        max=DEFAULT_RETRY_MAX_WAIT_SECONDS
    ),
    before_sleep=before_sleep,
    reraise=True
)

# Retry decorator for async functions


def api_retry_async(func):
    """
    Retry decorator for async functions.
    Handles both async and sync exceptions.
    """
    retry_attempts, retry_wait_seconds = get_retry_config()

    @wraps(func)
    async def wrapper(*args, **kwargs):
        attempt = 1
        while True:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt >= retry_attempts:
                    logger.error(
                        "Max retry attempts (%d) reached. Last error: %s",
                        retry_attempts, e
                    )
                    raise

                wait_time = min(
                    retry_wait_seconds * (2 ** (attempt - 1)),
                    DEFAULT_RETRY_MAX_WAIT_SECONDS
                )

                logger.warning(
                    "Retrying after error: %s. Attempt %d/%d. Waiting %d seconds...",
                    str(e), attempt, retry_attempts, wait_time
                )

                await asyncio.sleep(wait_time)
                attempt += 1

    return wrapper
