"""VS Code & Cursor Language Server Extension Scaffolder.

Generates the complete client configuration, package.json manifest, TextMate syntax grammar,
and Cursor rules (.cursorrules) to integrate PromptDiff LSP natively into modern AI editors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_vscode_manifest() -> dict[str, Any]:
    """Generate VS Code extension package.json manifest."""
    return {
        "name": "promptdiff-vscode",
        "displayName": "PromptDiff Language Tools",
        "description": "Language Server Protocol client for PromptDiff prompt engineering",
        "version": "3.4.0",
        "publisher": "promptdiff",
        "engines": {"vscode": "^1.80.0"},
        "categories": ["Programming Languages", "Linters"],
        "contributes": {
            "languages": [
                {
                    "id": "promptdiff",
                    "aliases": ["Prompt Template", "prompt"],
                    "extensions": [".prompt", ".prompt.txt"],
                }
            ],
            "configuration": {
                "title": "PromptDiff",
                "properties": {
                    "promptdiff.serverPath": {
                        "type": "string",
                        "default": "promptdiff",
                        "description": "Path to promptdiff executable",
                    }
                },
            },
        },
    }


def generate_cursor_rules() -> str:
    """Generate standard .cursorrules configuration for PromptDiff."""
    return """# PromptDiff Cursor Engineering Rules
- Always maintain Jinja2/bracket variable consistency in {{variables}}.
- Enforce strict JSON schema validation when prompting for structured data.
- Run `promptdiff check <prompt_path>` before committing prompt updates.
- Profile latency and token pricing with `promptdiff pricing <model>`.
"""


class ExtensionScaffolder:
    """Scaffolds editor configuration files for VS Code and Cursor."""

    def scaffold(self, output_dir: str = ".") -> dict[str, str]:
        """Generate editor integration files in output directory."""
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)

        manifest_path = target / "package.json"
        manifest_path.write_text(json.dumps(generate_vscode_manifest(), indent=2), encoding="utf-8")

        cursor_path = target / ".cursorrules"
        cursor_path.write_text(generate_cursor_rules(), encoding="utf-8")

        return {
            "vscode_manifest": str(manifest_path.resolve()),
            "cursor_rules": str(cursor_path.resolve()),
        }
