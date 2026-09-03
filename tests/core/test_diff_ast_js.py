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
