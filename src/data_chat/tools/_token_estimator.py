"""Token count estimation for tool response budget management.

Near-exact copy of datahub-agent-context/mcp_tools/_token_estimator.py.

WHY THIS EXISTS:
    Tools return data to an LLM. If the data is too large, it blows up the
    context window. We need to estimate token count BEFORE sending data back,
    so we can truncate if needed.

    Why not use a real tokenizer (tiktoken, etc.)?
    - Adds a heavy dependency
    - Slow for large payloads (tokenizers process char-by-char)
    - We don't need precision — DataHub uses a 90% budget buffer

    The heuristic: ~1.3 characters per token (accounts for JSON structural
    overhead like quotes, colons, commas that often tokenize as separate tokens).
"""

import logging
from typing import Union

logger = logging.getLogger(__name__)


class TokenCountEstimator:
    """Fast token estimation using character counting."""

    @staticmethod
    def estimate_dict_tokens(
        obj: Union[dict, list, str, int, float, bool, None],
    ) -> int:
        """Approximate token count for a dict/list structure.

        Recursively walks the structure counting characters.
        Protected against infinite recursion with MAX_DEPTH=100.

        Args:
            obj: Dict, list, or primitive (must not contain circular references)

        Returns:
            Approximate token count
        """
        MAX_DEPTH = 100

        def _count_chars(item, depth: int = 0) -> int:
            if depth > MAX_DEPTH:
                logger.error(
                    f"Max depth {MAX_DEPTH} exceeded in structure, stopping recursion"
                )
                return 0

            if item is None:
                return 4  # "null"
            elif isinstance(item, bool):
                return 5  # "true" or "false"
            elif isinstance(item, str):
                base_length = len(item)
                escape_overhead = int(base_length * 0.1)
                return base_length + 6 + escape_overhead  # +6 for quotes + structure
            elif isinstance(item, (int, float)):
                return 6  # average number length
            elif isinstance(item, list):
                return sum(_count_chars(elem, depth + 1) for elem in item) + len(item)
            elif isinstance(item, dict):
                total = 0
                for key, value in item.items():
                    total += len(str(key)) + 9  # "key": value, → quotes+colon+comma
                    total += _count_chars(value, depth + 1)
                return total + len(item)
            else:
                return 10  # fallback

        chars = _count_chars(obj, depth=0)
        return int(1.3 * chars / 4)
