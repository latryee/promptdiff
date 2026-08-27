"""Structural JSON Diffing Engine."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def compute_json_diff(obj1: Any, obj2: Any, path: str = "$") -> Dict[str, Any]:
    """Recursively compute structural differences between two JSON objects."""
    diff: Dict[str, Any] = {
        "added": {},
        "removed": {},
        "changed": {},
        "unchanged_keys": [],
    }

    if isinstance(obj1, dict) and isinstance(obj2, dict):
        keys1 = set(obj1.keys())
        keys2 = set(obj2.keys())

        # Added keys
        for k in keys2 - keys1:
            diff["added"][f"{path}.{k}"] = obj2[k]

        # Removed keys
        for k in keys1 - keys2:
            diff["removed"][f"{path}.{k}"] = obj1[k]

        # Common keys
        for k in keys1.intersection(keys2):
            v1, v2 = obj1[k], obj2[k]
            child_path = f"{path}.{k}"
            if v1 == v2:
                diff["unchanged_keys"].append(child_path)
            elif isinstance(v1, dict) and isinstance(v2, dict):
                nested = compute_json_diff(v1, v2, child_path)
                diff["added"].update(nested["added"])
                diff["removed"].update(nested["removed"])
                diff["changed"].update(nested["changed"])
                diff["unchanged_keys"].extend(nested["unchanged_keys"])
            else:
                diff["changed"][child_path] = {
                    "v1": v1,
                    "v2": v2,
                    "type_v1": type(v1).__name__,
                    "type_v2": type(v2).__name__,
                }
    else:
        if obj1 != obj2:
            diff["changed"][path] = {
                "v1": obj1,
                "v2": obj2,
                "type_v1": type(obj1).__name__,
                "type_v2": type(obj2).__name__,
            }

    return diff
