"""Unit tests for Word Diff and JSON Diff Engines."""

from promptdiff.diff.json_diff import compute_json_diff
from promptdiff.diff.text_diff import compute_line_diff, compute_word_diff


def test_compute_word_diff():
    text1 = "The quick brown fox"
    text2 = "The fast brown fox"
    chunks = compute_word_diff(text1, text2)

    assert len(chunks) > 0
    kinds = [c.kind for c in chunks]
    assert "equal" in kinds
    # 'quick' -> 'fast' is a replace or delete+insert
    assert ("replace" in kinds) or ("delete" in kinds and "insert" in kinds)


def test_compute_line_diff():
    text1 = "line1\nline2\nline3"
    text2 = "line1\nline_modified\nline3"
    lines = compute_line_diff(text1, text2)
    assert len(lines) >= 3


def test_compute_json_diff():
    obj1 = {
        "user_id": 123,
        "name": "Alice",
        "settings": {"dark_mode": True, "notifications": "all"},
        "status": "pending",
    }
    obj2 = {
        "user_id": 123,
        "name": "Alice Cooper",
        "settings": {"dark_mode": True, "notifications": "mentions", "new_opt": 1},
        "status": "active",
    }

    diff = compute_json_diff(obj1, obj2)
    assert "$.name" in diff["changed"]
    assert "$.status" in diff["changed"]
    assert "$.settings.notifications" in diff["changed"]
    assert "$.settings.new_opt" in diff["added"]
    assert "$.user_id" in diff["unchanged_keys"]
