"""Unit tests for JavaScript and TypeScript AST structural diffing."""

from __future__ import annotations

from promptdiff.diff.ast_diff import ASTStructuredDiffer


def test_diff_javascript_identical() -> None:
    differ = ASTStructuredDiffer()
    js1 = """
    // Calculate total price
    function calculateTotal(price, tax) {
        const total = price + tax;
        return total;
    }
    """
    js2 = """
    function calculateTotal(price, tax) {
        /* comment changed */
        const total = price + tax;
        return total;
    }
    """
    res = differ.diff_javascript(js1, js2)
    assert res.is_identical is True
    assert res.tree_edit_distance == 0


def test_diff_javascript_structural_change() -> None:
    differ = ASTStructuredDiffer()
    js1 = """
    function fetchUserData(userId) {
        return api.get("/users/" + userId);
    }
    """
    js2 = """
    async function fetchUserData(userId, options) {
        return api.get("/users/" + userId);
    }
    class UserService {}
    """
    res = differ.diff_javascript(js1, js2)
    assert res.is_identical is False
    assert res.tree_edit_distance > 0
    # Class added
    assert any(d.change_type == "ADDED" and "UserService" in d.path for d in res.differences)


def test_diff_typescript_signatures() -> None:
    differ = ASTStructuredDiffer()
    ts1 = """
    import { User } from "./models";
    function processUser(user: User): boolean {
        return user.isActive;
    }
    """
    ts2 = """
    import { User, Admin } from "./models";
    function processUser(user: User, admin: Admin): boolean {
        return user.isActive;
    }
    """
    res = differ.diff_javascript(ts1, ts2)
    assert res.is_identical is False
    assert res.tree_edit_distance > 0


def test_diff_ast_list_multiple_elements_added_removed() -> None:
    """Verify that adding multiple (2+) elements to a list reports each element individually as ADDED or REMOVED."""
    differ = ASTStructuredDiffer()

    # JSON lists with 2+ added elements
    j1 = '{"items": [{"id": {"name": "item1"}}]}'
    j2 = '{"items": [{"id": {"name": "item1"}}, {"id": {"name": "item2"}}, {"id": {"name": "item3"}}]}'

    res_add = differ.diff_json(j1, j2)
    assert res_add.is_identical is False
    # len changed diff should be preserved
    assert any(d.change_type == "VALUE_CHANGED" and "len=1" in str(d.v1_value) for d in res_add.differences)
    # item2 and item3 both individually reported as ADDED
    added_paths = [d.path for d in res_add.differences if d.change_type == "ADDED"]
    assert any("item2" in p for p in added_paths)
    assert any("item3" in p for p in added_paths)

    # Reverse direction: 2+ removed elements
    res_rem = differ.diff_json(j2, j1)
    assert res_rem.is_identical is False
    removed_paths = [d.path for d in res_rem.differences if d.change_type == "REMOVED"]
    assert any("item2" in p for p in removed_paths)
    assert any("item3" in p for p in removed_paths)

    # JavaScript code with 2+ added functions/classes
    code1 = "function first() { return 1; }"
    code2 = "function first() { return 1; }\nfunction second() { return 2; }\nclass ThirdService {}"
    res_js = differ.diff_javascript(code1, code2)
    assert res_js.is_identical is False
    js_added = [d for d in res_js.differences if d.change_type == "ADDED"]
    assert len(js_added) >= 2
    assert any("second" in d.path or (isinstance(d.v2_value, dict) and "second" in str(d.v2_value)) for d in js_added)
    assert any("ThirdService" in d.path for d in js_added)
