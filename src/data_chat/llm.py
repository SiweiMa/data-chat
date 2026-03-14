"""LLM client factory for data-chat agents.

Extracted from examples/raw_api/basic_agent.py to be reusable across
the CLI example, Streamlit app, and any future interfaces.

Priority for LLM backend:
1. Explicit params (api_key, base_url, model) — full control
2. SoFi LLM Proxy — auto-detected via llm-proxy-keys CLI
3. ANTHROPIC_API_KEY env var — direct Anthropic API
"""

import logging
import os
import subprocess
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)

# SoFi LLM Proxy config
LLM_PROXY_BASE_URL = "https://internal.sofitest.com/llm-proxy"
LLM_PROXY_KEYS_CLI = os.path.expanduser("~/.local/bin/llm-proxy-keys")
LLM_PROXY_MODEL = "claude-sonnet-4-6"

# Direct Anthropic API config (fallback)
DIRECT_MODEL = "claude-sonnet-4-20250514"


def _get_proxy_api_key() -> Optional[str]:
    """Get an API key from SoFi's llm-proxy-keys CLI.

    The CLI caches keys and handles Okta authentication automatically.
    Returns None if the CLI is not installed or fails.
    """
    if not os.path.exists(LLM_PROXY_KEYS_CLI):
        return None

    try:
        result = subprocess.run(
            [LLM_PROXY_KEYS_CLI, "--no-refresh", "--quiet"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        key = result.stdout.strip()
        if key and key.startswith("sk-"):
            return key
    except Exception:
        pass

    # Try with refresh (may open browser for Okta)
    try:
        result = subprocess.run(
            [LLM_PROXY_KEYS_CLI, "--quiet"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        key = result.stdout.strip()
        if key and key.startswith("sk-"):
            return key
    except Exception:
        pass

    return None


def create_anthropic_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> tuple[anthropic.Anthropic, str]:
    """Create an Anthropic client with auto-detection of backend.

    Priority:
    1. Explicit params — if api_key is provided, use it directly
    2. SoFi LLM Proxy — if llm-proxy-keys CLI is available
    3. ANTHROPIC_API_KEY env var — direct Anthropic API

    Args:
        api_key: Explicit API key (skips auto-detection)
        base_url: Explicit base URL (e.g., for a proxy)
        model: Explicit model name (overrides default)

    Returns:
        Tuple of (anthropic.Anthropic client, model name string)

    Raises:
        RuntimeError: If no API key can be found
    """
    # Option 1: Explicit params
    if api_key:
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        resolved_model = model or DIRECT_MODEL
        logger.info("Using explicit API key, model: %s", resolved_model)
        return anthropic.Anthropic(**client_kwargs), resolved_model

    # Option 2: SoFi LLM Proxy
    proxy_key = _get_proxy_api_key()
    if proxy_key:
        resolved_model = model or LLM_PROXY_MODEL
        logger.info("Using SoFi LLM Proxy: %s, model: %s", LLM_PROXY_BASE_URL, resolved_model)
        client = anthropic.Anthropic(
            api_key=proxy_key,
            base_url=base_url or LLM_PROXY_BASE_URL,
        )
        return client, resolved_model

    # Option 3: Direct Anthropic API
    env_key = os.environ.get("ANTHROPIC_API_KEY")
    if env_key:
        resolved_model = model or DIRECT_MODEL
        logger.info("Using direct Anthropic API, model: %s", resolved_model)
        client = anthropic.Anthropic(api_key=env_key)
        return client, resolved_model

    raise RuntimeError(
        "No LLM API key found. Either:\n"
        "  - Install llm-proxy-keys CLI (SoFi employees)\n"
        "  - Set ANTHROPIC_API_KEY environment variable\n"
        "  - Pass api_key= explicitly"
    )
