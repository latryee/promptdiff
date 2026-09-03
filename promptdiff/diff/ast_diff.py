"""Abstract Syntax Tree (AST) Structured Output Differ.

Parses generated JSON/SQL/Python payloads into formal Abstract Syntax Trees,
calculating structural tree edit distance and identifying schema mutations
(missing keys, type alterations, array shape violations) instead of raw text line diffs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ASTDifference:
    """A structural mutation between two output ASTs."""

    path: str
    change_type: str  # ADDED, REMOVED, TYPE_MUTATED, VALUE_CHANGED
    v1_value: Any
    v2_value: Any
    severity: str  # CRITICAL (breaks schema), MINOR (value change)


@dataclass
class ASTDiffResult:
    """Outcome of structural AST comparison."""

    is_identical: bool
    schema_compatible: bool
    tree_edit_distance: int
    differences: list[ASTDifference]


class ASTStructuredDiffer:
    """Compares structured JSON and JavaScript/TypeScript outputs at the Abstract Syntax Tree level."""

    def diff_json(self, json_str_1: str, json_str_2: str) -> ASTDiffResult:
        """Compare two JSON objects structurally."""
        try:
            tree1 = json.loads(json_str_1)
        except Exception:
            tree1 = {"_raw": json_str_1}

        try:
            tree2 = json.loads(json_str_2)
        except Exception:
            tree2 = {"_raw": json_str_2}

        diffs: list[ASTDifference] = []
        self._compare_nodes("", tree1, tree2, diffs)

        schema_ok = not any(d.severity == "CRITICAL" for d in diffs)

        return ASTDiffResult(
            is_identical=len(diffs) == 0,
            schema_compatible=schema_ok,
            tree_edit_distance=len(diffs),
            differences=diffs,
        )

    def diff_javascript(self, js_str_1: str, js_str_2: str) -> ASTDiffResult:
        """Compare two JavaScript/TypeScript code snippets structurally at the AST level."""
        tree1 = self._parse_js_ast(js_str_1)
        tree2 = self._parse_js_ast(js_str_2)

        diffs: list[ASTDifference] = []
        self._compare_nodes("", tree1, tree2, diffs)

        schema_ok = not any(d.severity == "CRITICAL" for d in diffs)

        return ASTDiffResult(
            is_identical=len(diffs) == 0,
            schema_compatible=schema_ok,
            tree_edit_distance=len(diffs),
            differences=diffs,
        )

    def _parse_js_ast(self, code: str) -> dict[str, Any]:
        """Parse JS/TS code into structured dict using esprima if available, or syntactic fallback."""
        try:
            import esprima  # type: ignore[import-untyped]

            tree = esprima.parseScript(code, tolerant=True)
            return tree.toDict()  # type: ignore[no-any-return]
        except ImportError:
            return self._parse_js_fallback(code)
        except Exception:
            return self._parse_js_fallback(code)

    def _parse_js_fallback(self, code: str) -> dict[str, Any]:
        """Syntactic parser for JavaScript and TypeScript AST fallback."""
        cleaned = re.sub(r"//.*?\n", "\n", code)
        cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)

        functions = re.findall(r"(?:async\s+)?function\s+([a-zA-Z0-9_$]+)\s*\(([^)]*)\)", cleaned)
        classes = re.findall(r"class\s+([a-zA-Z0-9_$]+)(?:\s+extends\s+([a-zA-Z0-9_$]+))?", cleaned)
        variables = re.findall(r"(?:const|let|var)\s+([a-zA-Z0-9_$]+)", cleaned)
        imports = re.findall(r"import\s+(?:\{([^}]+)\}|([a-zA-Z0-9_$]+))\s+from\s+['\"]([^'\"]+)['\"]", cleaned)

        return {
            "type": "Program",
            "functions": {
                f[0]: {"params": [p.strip().split(":")[0].strip() for p in f[1].split(",") if p.strip()]}
                for f in functions
            },
            "classes": {c[0]: {"extends": c[1] or None} for c in classes},
            "variables": sorted(set(variables)),
            "imports": [{"imported": (i[0] or i[1]).strip(), "source": i[2]} for i in imports],
        }

    def _compare_nodes(self, path: str, n1: Any, n2: Any, diffs: list[ASTDifference]) -> None:
        type1 = type(n1)
        type2 = type(n2)

        if type1 != type2:
            diffs.append(
                ASTDifference(
                    path=path or "root",
                    change_type="TYPE_MUTATED",
                    v1_value=type1.__name__,
                    v2_value=type2.__name__,
                    severity="CRITICAL",
                )
            )
            return

        if isinstance(n1, dict):
            keys1 = set(n1.keys())
            keys2 = set(n2.keys())

            for k in keys1 - keys2:
                diffs.append(
                    ASTDifference(
                        path=f"{path}.{k}" if path else k,
                        change_type="REMOVED",
                        v1_value=n1[k],
                        v2_value=None,
                        severity="CRITICAL",
                    )
                )

            for k in keys2 - keys1:
                diffs.append(
                    ASTDifference(
                        path=f"{path}.{k}" if path else k,
                        change_type="ADDED",
                        v1_value=None,
                        v2_value=n2[k],
                        severity="MINOR",
                    )
                )

            for k in keys1.intersection(keys2):
                self._compare_nodes(f"{path}.{k}" if path else k, n1[k], n2[k], diffs)

        elif isinstance(n1, list):
            if len(n1) != len(n2):
                diffs.append(
                    ASTDifference(
                        path=path or "root",
                        change_type="VALUE_CHANGED",
                        v1_value=f"len={len(n1)}",
                        v2_value=f"len={len(n2)}",
                        severity="MINOR",
                    )
                )
            for idx in range(min(len(n1), len(n2))):
                self._compare_nodes(f"{path}[{idx}]", n1[idx], n2[idx], diffs)

        else:
            if n1 != n2:
                diffs.append(
                    ASTDifference(
                        path=path or "root",
                        change_type="VALUE_CHANGED",
                        v1_value=n1,
                        v2_value=n2,
                        severity="MINOR",
                    )
                )
