"""Tests for helper utilities (token budget, sanitization, truncation).

Tests the essential LLM agent tricks from helpers.py:
1. Two-level token budget (per-table + per-response)
2. Truncation metadata flags
3. Sanitize before truncate
4. ReDoS protection
5. Cell value truncation
"""

import pytest

from data_chat.tools._token_estimator import TokenCountEstimator
from data_chat.tools.helpers import (
    sanitize_html_content,
    sanitize_and_truncate,
    truncate_descriptions,
    truncate_query,
    truncate_cell_values,
    truncate_with_ellipsis,
    select_results_within_budget,
    select_columns_within_budget,
)


# --- TokenCountEstimator ---


def test_estimator_small_vs_large():
    """Larger structures should have higher token estimates."""
    small = {"name": "users"}
    large = {"columns": [{"name": f"col_{i}", "type": "VARCHAR"} for i in range(100)]}
    assert TokenCountEstimator.estimate_dict_tokens(small) < TokenCountEstimator.estimate_dict_tokens(large)


def test_estimator_handles_primitives():
    """Should handle None, bool, str, int, float."""
    assert TokenCountEstimator.estimate_dict_tokens(None) > 0
    assert TokenCountEstimator.estimate_dict_tokens(True) > 0
    assert TokenCountEstimator.estimate_dict_tokens("hello") > 0
    assert TokenCountEstimator.estimate_dict_tokens(42) > 0
    assert TokenCountEstimator.estimate_dict_tokens(3.14) > 0


# --- sanitize_html_content ---


def test_sanitize_strips_html_tags():
    """Should remove HTML tags and decode entities."""
    assert "actual" in sanitize_html_content("<b>actual</b> content")
    assert "<" not in sanitize_html_content("<div>text</div>")


def test_sanitize_decodes_entities():
    """Should decode &amp;, &lt;, etc."""
    assert "&" in sanitize_html_content("a &amp; b")


def test_sanitize_redos_protection():
    """Bounded regex should not hang on malicious input."""
    import time
    malicious = "<" + "a" * 100000
    start = time.time()
    sanitize_html_content(malicious)
    assert time.time() - start < 1.0


def test_sanitize_empty():
    """Should handle empty/None input."""
    assert sanitize_html_content("") == ""
    assert sanitize_html_content(None) is None


# --- sanitize_and_truncate ---


def test_sanitize_before_truncate():
    """Should strip HTML first, then truncate (order matters)."""
    html = '<div class="wrapper"><p>This is content.</p></div>'
    result = sanitize_and_truncate(html, 50)
    assert "<" not in result
    assert "content" in result


# --- truncate_descriptions ---


def test_truncate_descriptions_in_place():
    """Should truncate 'description' and 'comment' keys recursively."""
    data = {"comment": "x" * 2000, "nested": {"description": "y" * 2000}}
    truncate_descriptions(data)
    assert len(data["comment"]) <= 1000
    assert len(data["nested"]["description"]) <= 1000


def test_truncate_descriptions_leaves_short_values():
    """Should not modify values under the limit."""
    data = {"comment": "short"}
    truncate_descriptions(data)
    assert data["comment"] == "short"


# --- truncate_query ---


def test_truncate_query_uses_explicit_suffix():
    """Should use '... [truncated]' suffix, not just '...'."""
    long_sql = "SELECT " + ", ".join([f"col_{i}" for i in range(1000)])
    result = truncate_query(long_sql)
    assert result.endswith("... [truncated]")
    assert len(result) <= 5000


def test_truncate_query_short_passthrough():
    """Short SQL should pass through unchanged."""
    short = "SELECT 1"
    assert truncate_query(short) == short


# --- truncate_cell_values ---


def test_truncate_cell_values_large_string():
    """Should truncate oversized string cells."""
    row = {"id": 1, "json_blob": "x" * 5000}
    result = truncate_cell_values(row)
    assert len(result["json_blob"]) <= 2000
    assert result["json_blob"].endswith("... [truncated]")
    assert result["id"] == 1  # non-strings pass through


def test_truncate_cell_values_small_passthrough():
    """Small values should pass through unchanged."""
    row = {"name": "alice", "age": 30}
    result = truncate_cell_values(row)
    assert result == row


# --- select_results_within_budget ---


def test_budget_truncates_at_limit():
    """Should stop yielding when token budget would be exceeded."""
    results = [{"data": "x" * 1000} for _ in range(50)]
    selected = list(select_results_within_budget(
        iter(results), fetch_entity=lambda r: r, max_results=50, token_budget=5000,
    ))
    assert 1 <= len(selected) < 50


def test_budget_yields_at_least_one():
    """Should always yield at least 1 result even if it exceeds budget."""
    results = [{"data": "x" * 10000}]  # single huge result
    selected = list(select_results_within_budget(
        iter(results), fetch_entity=lambda r: r, max_results=10, token_budget=100,
    ))
    assert len(selected) == 1


def test_budget_respects_max_results():
    """Should not exceed max_results even if budget allows."""
    results = [{"x": 1} for _ in range(100)]
    selected = list(select_results_within_budget(
        iter(results), fetch_entity=lambda r: r, max_results=5,
    ))
    assert len(selected) == 5


# --- select_columns_within_budget ---


def test_columns_budget_truncates_wide_table():
    """Wide table should be truncated with columns_truncated flag."""
    columns = [{
        "name": f"col_{i}", "type": "VARCHAR(16777216)",
        "comment": f"Description for column {i} with some extra text for padding.",
        "position": i,
    } for i in range(500)]

    result = select_columns_within_budget(columns)
    assert result["columns_truncated"] is True
    assert result["returned"] < 500
    assert result["total_columns"] == 500


def test_columns_budget_narrow_table_no_flag():
    """Narrow table should NOT have columns_truncated flag."""
    columns = [{"name": f"col_{i}", "type": "INT"} for i in range(5)]
    result = select_columns_within_budget(columns)
    assert "columns_truncated" not in result
    assert result["returned"] == 5


def test_columns_budget_pagination():
    """offset and limit should work for pagination."""
    columns = [{"name": f"col_{i}", "type": "INT"} for i in range(20)]
    result = select_columns_within_budget(columns, offset=5, limit=3)
    assert result["returned"] == 3
    assert result["offset"] == 5
    assert result["columns_truncated"] is True
