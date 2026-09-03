"""Configuration & Dataset Loader for promptdiff.

Loads prompt files, testcase datasets (JSONL, JSON, YAML, CSV), and config files.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from promptdiff.core.models import PromptVersion, TestCase

logger = logging.getLogger("promptdiff.core.config")


class ProjectConfig(BaseModel):
    """Configuration structure loaded from promptdiff.yaml."""

    v1_prompt: str | None = None
    v2_prompt: str | None = None
    model: str = "gpt-4o"
    temperature: float = 0.0
    evaluators: list[str] = Field(default_factory=lambda: ["json_validity", "latency", "cost", "similarity"])
    assertions: list[str] = Field(default_factory=list)
    dataset: str | None = None
    concurrency: int = 4
    cache: bool = True


def load_prompt_file(
    file_path: str, version_name: str = "v1", model: str = "gpt-4o", temperature: float = 0.0
) -> PromptVersion:
    """Load prompt template from a local file or string.

    Args:
        file_path: Path to prompt file (.txt, .md, .prompt, etc.) or raw string.
        version_name: Identifier name (v1, v2).
        model: Default model.
        temperature: Sampling temperature.

    Returns:
        PromptVersion model.
    """
    path_obj = Path(file_path)
    if path_obj.is_file():
        content = path_obj.read_text(encoding="utf-8")
        return PromptVersion(
            name=version_name,
            path=str(path_obj.resolve()),
            template=content,
            model=model,
            temperature=temperature,
        )
    else:
        has_path_separator = "/" in file_path or "\\" in file_path
        has_file_extension = file_path.lower().endswith(
            (".txt", ".md", ".prompt", ".yaml", ".yml", ".json", ".jinja", ".j2")
        )
        if has_path_separator or has_file_extension:
            logger.warning(f"Path '{file_path}' looks like a file path but not found, treating as literal prompt")

        # Treat as inline prompt string
        return PromptVersion(
            name=version_name,
            template=file_path,
            model=model,
            temperature=temperature,
        )


def load_dataset(dataset_path: str | None) -> list[TestCase]:
    """Load test cases from JSONL, JSON, YAML, or CSV files.

    Args:
        dataset_path: Path to dataset file.

    Returns:
        List of TestCase objects. Defaults to a single empty TestCase if None.
    """
    if not dataset_path:
        return [TestCase(id="default_case", description="Default single execution", vars={})]

    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    suffix = path.suffix.lower()
    test_cases: list[TestCase] = []

    if suffix in [".jsonl", ".ndjson"]:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                test_cases.append(_parse_testcase_dict(data, f"case_{i + 1}"))

    elif suffix == ".json":
        with open(path, encoding="utf-8") as f:
            raw_data = json.load(f)
            if isinstance(raw_data, list):
                for i, item in enumerate(raw_data):
                    test_cases.append(_parse_testcase_dict(item, f"case_{i + 1}"))
            elif isinstance(raw_data, dict):
                test_cases.append(_parse_testcase_dict(raw_data, "case_1"))

    elif suffix in [".yaml", ".yml"]:
        with open(path, encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
            if isinstance(raw_data, list):
                for i, item in enumerate(raw_data):
                    test_cases.append(_parse_testcase_dict(item, f"case_{i + 1}"))
            elif isinstance(raw_data, dict):
                cases = raw_data.get("testcases", raw_data.get("tests", [raw_data]))
                if isinstance(cases, list):
                    for i, item in enumerate(cases):
                        test_cases.append(_parse_testcase_dict(item, f"case_{i + 1}"))

    elif suffix == ".csv":
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                test_cases.append(
                    TestCase(
                        id=row.get("id", f"case_{i + 1}"),
                        description=row.get("description", ""),
                        vars=dict(row),
                        expected_output=row.get("expected_output"),
                    )
                )
    else:
        raise ValueError(f"Unsupported dataset format: {suffix}. Supported formats: .jsonl, .json, .yaml, .csv")

    return test_cases


def _parse_testcase_dict(data: dict[str, Any], default_id: str) -> TestCase:
    """Helper to convert dictionary structure into TestCase object."""
    tc_id = data.get("id", default_id)
    description = data.get("description", "")
    expected = data.get("expected_output", data.get("expected", None))
    schema = data.get("schema", None)
    tags = data.get("tags", [])

    # If variables are wrapped under 'vars' or 'inputs'
    if "vars" in data and isinstance(data["vars"], dict):
        variables = data["vars"]
    elif "inputs" in data and isinstance(data["inputs"], dict):
        variables = data["inputs"]
    else:
        # Treat all top-level keys except metadata as variables
        variables = {
            k: v
            for k, v in data.items()
            if k not in {"id", "description", "expected_output", "expected", "schema", "tags"}
        }

    return TestCase(
        id=str(tc_id),
        description=description,
        vars=variables,
        expected_output=expected,
        schema=schema,
        tags=tags,
    )


def load_project_config(config_path: str | None = None) -> ProjectConfig:
    """Load configuration from promptdiff.yaml or defaults."""
    target = Path(config_path) if config_path else Path("promptdiff.yaml")
    if target.is_file():
        with open(target, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return ProjectConfig.model_validate(data)
    return ProjectConfig()
