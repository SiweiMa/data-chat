"""Structured exception hierarchy for data-chat.

Only includes exceptions that are actually raised somewhere in the codebase.
Each exception carries enough context for the UI to display a helpful message.
"""

from typing import Optional


class DataChatError(Exception):
    """Base exception for all data-chat errors."""

    pass


class SnowflakeConnectionError(DataChatError):
    """Raised after repeated Snowflake connection failures during tool dispatch."""

    pass


class SnowflakeSessionExpiredError(DataChatError):
    """Raised when the Snowflake session has timed out."""

    pass


class LLMConnectionError(DataChatError):
    """Raised when the LLM API is unreachable."""

    pass


class LLMRateLimitError(DataChatError):
    """Raised when the LLM API returns 429 (rate limited).

    Attributes:
        retry_after: Seconds to wait before retrying (from Retry-After header).
    """

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMAPIError(DataChatError):
    """Raised when the LLM API returns a server error (500, 503, etc.).

    Attributes:
        status_code: HTTP status code from the API response.
    """

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code
