"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

from promptdiff.diff.ast_diff import ASTStructuredDiffer


def test_ast_diff() -> None:
    """Test structural AST diffing of JSON outputs."""
    differ = ASTStructuredDiffer()
    json1 = '{"user": {"name": "Alice", "age": 30}, "active": true}'
    json2 = '{"user": {"name": "Alice", "age": "thirty"}, "role": "admin"}'

    res = differ.diff_json(json1, json2)
    assert res.is_identical is False
    assert res.tree_edit_distance >= 2
    assert any(d.change_type == "TYPE_MUTATED" for d in res.differences)
