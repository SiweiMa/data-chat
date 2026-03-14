"""Tests for LLM client factory."""

import os
from unittest.mock import patch

import pytest


def test_create_anthropic_client_with_explicit_key():
    """Explicit api_key should be used directly."""
    from data_chat.llm import create_anthropic_client

    client, model = create_anthropic_client(api_key="sk-test-key")
    assert client.api_key == "sk-test-key"
    # Default model when explicit key is used
    assert "claude" in model.lower() or "sonnet" in model.lower()


def test_create_anthropic_client_env_fallback():
    """Should fall back to ANTHROPIC_API_KEY env var."""
    from data_chat.llm import create_anthropic_client

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-env-key"}):
        # Ensure proxy is not available
        with patch("data_chat.llm._get_proxy_api_key", return_value=None):
            client, model = create_anthropic_client()

    assert client.api_key == "sk-env-key"


def test_create_anthropic_client_no_key_raises():
    """Should raise RuntimeError when no API key is available."""
    from data_chat.llm import create_anthropic_client

    with patch.dict(os.environ, {}, clear=True):
        # Remove ANTHROPIC_API_KEY if present
        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            with patch("data_chat.llm._get_proxy_api_key", return_value=None):
                with pytest.raises(RuntimeError, match="No LLM API key"):
                    create_anthropic_client()
