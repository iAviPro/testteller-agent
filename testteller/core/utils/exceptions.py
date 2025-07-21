"""
Custom exceptions for the TestTeller agent.
"""


class EmbeddingGenerationError(Exception):
    """Custom exception for errors during embedding generation."""

    def __init__(self, message="Failed to generate embeddings", provider=None, original_exception=None):
        self.provider = provider
        self.original_exception = original_exception

        full_message = f"[{provider.upper() if provider else 'LLM'} SERVICE]" \
            f" - {message}"

        if original_exception:
            # Try to extract a more specific error message from the original exception
            # For example, from OpenAI's RateLimitError
            if hasattr(original_exception, 'message'):
                # Handle cases where message might be bytes
                try:
                    error_detail = original_exception.message.decode() if isinstance(
                        original_exception.message, bytes) else str(original_exception.message)
                    full_message += f"\n  - Root Cause: {error_detail}"
                except (UnicodeDecodeError, AttributeError):
                    full_message += f"\n  - Root Cause: {original_exception}"
            else:
                full_message += f"\n  - Root Cause: {original_exception}"

        super().__init__(full_message)


class DocumentIngestionError(Exception):
    """Custom exception for errors during document ingestion."""
    pass


class CodeIngestionError(Exception):
    """Custom exception for errors during code ingestion."""
    pass


class TestCaseGenerationError(Exception):
    """Custom exception for errors during test case generation."""
    __test__ = False  # Tell pytest this is not a test class
    pass
