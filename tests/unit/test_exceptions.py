"""Tests for the exception hierarchy."""

from data_chat.exceptions import (
    DataChatError,
    LLMAPIError,
    LLMConnectionError,
    LLMRateLimitError,
    SnowflakeConnectionError,
    SnowflakeSessionExpiredError,
)


def test_exception_hierarchy():
    """All exceptions should inherit from DataChatError."""
    assert issubclass(SnowflakeConnectionError, DataChatError)
    assert issubclass(SnowflakeSessionExpiredError, DataChatError)
    assert issubclass(LLMConnectionError, DataChatError)
    assert issubclass(LLMRateLimitError, DataChatError)
    assert issubclass(LLMAPIError, DataChatError)

    # All should also be Exceptions
    assert issubclass(DataChatError, Exception)


def test_llm_rate_limit_retry_after():
    """LLMRateLimitError should carry retry_after."""
    err = LLMRateLimitError("rate limited", retry_after=30.0)
    assert err.retry_after == 30.0
    assert "rate limited" in str(err)

    # None is valid
    err2 = LLMRateLimitError("rate limited")
    assert err2.retry_after is None


def test_llm_api_error_status_code():
    """LLMAPIError should carry status_code."""
    err = LLMAPIError("server error", status_code=503)
    assert err.status_code == 503
    assert "server error" in str(err)

    # None is valid
    err2 = LLMAPIError("unknown error")
    assert err2.status_code is None
