"""Configuration & Dataset Loader for promptdiff.

Loads prompt files, testcase datasets (JSONL, JSON, YAML, CSV), and config files.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from promptdiff.core.exceptions import DatasetLoadError
from promptdiff.core.models import PromptVersion, TestCase

logger = logging.getLogger("promptdiff.core.config")


class DatasetError(DatasetLoadError, ValueError):
    """Raised when a dataset file cannot be loaded, parsed, or validated."""

    def __init__(
        self,
        message: str,
        line_number: int | None = None,
        file_path: str | None = None,
    ) -> None:
        self.line_number = line_number
        self.file_path = file_path
        loc = f" at line {line_number}" if line_number is not None else ""
        src = f" in '{file_path}'" if file_path else ""
        super().__init__(f"Dataset error{src}{loc}: {message}")


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
    timeout: float | None = None
    cache_ttl: int | None = None
    tags: list[str] | None = None
    limit: int | None = None
    redact: bool = False
    experiment_id: str | None = None


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


def load_dataset(
    dataset_path: str | None,
    tags: list[str] | None = None,
    limit: int | None = None,
) -> list[TestCase]:
    """Load test cases from JSONL, JSON, YAML, or CSV files.

    Args:
        dataset_path: Path to dataset file.
        tags: Optional filter to only include test cases matching at least one tag.
        limit: Optional maximum number of test cases to return.

    Returns:
        List of TestCase objects. Defaults to a single empty TestCase if None.
    """
    if not dataset_path:
        return [TestCase(id="default_case", description="Default single execution", vars={})]

    return list(stream_dataset(dataset_path, tags=tags, limit=limit))


def stream_dataset(
    dataset_path: str,
    tags: list[str] | None = None,
    limit: int | None = None,
) -> Iterator[TestCase]:
    """Stream test cases lazily from file without keeping all parsed structures in memory.

    Args:
        dataset_path: Path to dataset file.
        tags: Optional filter to only include test cases matching at least one tag.
        limit: Optional maximum number of test cases to yield.

    Yields:
        TestCase objects.
    """
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    suffix = path.suffix.lower()
    yielded_count = 0
    seen_ids: set[str] = set()

    def _should_include(tc: TestCase) -> bool:
        if tags and not any(t in tc.tags for t in tags):
            return False
        return True

    if suffix in [".jsonl", ".ndjson"]:
        with open(path, encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                except json.JSONDecodeError as err:
                    raise DatasetError(
                        f"Malformed JSON syntax: {err.msg}",
                        line_number=line_idx,
                        file_path=str(path),
                    ) from err
                if not isinstance(data, dict):
                    raise DatasetError(
                        f"Line must contain a JSON object, got {type(data).__name__}",
                        line_number=line_idx,
                        file_path=str(path),
                    )

                tc = _parse_testcase_dict(data, f"case_{line_idx}")
                if tc.id in seen_ids:
                    logger.warning("Duplicate testcase ID '%s' found at line %d in '%s'", tc.id, line_idx, path)
                seen_ids.add(tc.id)

                if _should_include(tc):
                    yield tc
                    yielded_count += 1
                    if limit is not None and yielded_count >= limit:
                        return

    elif suffix == ".json":
        with open(path, encoding="utf-8") as f:
            try:
                raw_data = json.load(f)
            except json.JSONDecodeError as err:
                raise DatasetError(
                    f"Malformed JSON syntax: {err.msg}",
                    line_number=err.lineno,
                    file_path=str(path),
                ) from err

            items: list[dict[str, Any]]
            if isinstance(raw_data, list):
                items = raw_data
            elif isinstance(raw_data, dict):
                cases_list = raw_data.get("testcases", raw_data.get("tests"))
                items = cases_list if isinstance(cases_list, list) else [raw_data]
            else:
                raise DatasetError(
                    f"Top-level JSON must be a list or object, got {type(raw_data).__name__}",
                    file_path=str(path),
                )

            for i, item in enumerate(items, start=1):
                if not isinstance(item, dict):
                    raise DatasetError(
                        f"Item {i} in JSON array must be an object, got {type(item).__name__}",
                        line_number=i,
                        file_path=str(path),
                    )
                tc = _parse_testcase_dict(item, f"case_{i}")
                if tc.id in seen_ids:
                    logger.warning("Duplicate testcase ID '%s' found in '%s'", tc.id, path)
                seen_ids.add(tc.id)

                if _should_include(tc):
                    yield tc
                    yielded_count += 1
                    if limit is not None and yielded_count >= limit:
                        return

    elif suffix in [".yaml", ".yml"]:
        with open(path, encoding="utf-8") as f:
            try:
                raw_data = yaml.safe_load(f)
            except yaml.YAMLError as err:
                lineno = getattr(getattr(err, "problem_mark", None), "line", None)
                if lineno is not None:
                    lineno += 1
                raise DatasetError(
                    f"Malformed YAML syntax: {err}",
                    line_number=lineno,
                    file_path=str(path),
                ) from err

            items_yaml: list[dict[str, Any]]
            if isinstance(raw_data, list):
                items_yaml = raw_data
            elif isinstance(raw_data, dict):
                cases_list = raw_data.get("testcases", raw_data.get("tests", [raw_data]))
                items_yaml = cases_list if isinstance(cases_list, list) else [raw_data]
            else:
                raise DatasetError(
                    f"Top-level YAML must be a list or mapping, got {type(raw_data).__name__}",
                    file_path=str(path),
                )

            for i, item in enumerate(items_yaml, start=1):
                if not isinstance(item, dict):
                    raise DatasetError(
                        f"Item {i} in YAML list must be a mapping, got {type(item).__name__}",
                        line_number=i,
                        file_path=str(path),
                    )
                tc = _parse_testcase_dict(item, f"case_{i}")
                if tc.id in seen_ids:
                    logger.warning("Duplicate testcase ID '%s' found in '%s'", tc.id, path)
                seen_ids.add(tc.id)

                if _should_include(tc):
                    yield tc
                    yielded_count += 1
                    if limit is not None and yielded_count >= limit:
                        return

    elif suffix == ".csv":
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, start=1):
                tc = TestCase(
                    id=row.get("id", f"case_{i}"),
                    description=row.get("description", ""),
                    vars=dict(row),
                    expected_output=row.get("expected_output"),
                )
                if tc.id in seen_ids:
                    logger.warning("Duplicate testcase ID '%s' found in '%s'", tc.id, path)
                seen_ids.add(tc.id)

                if _should_include(tc):
                    yield tc
                    yielded_count += 1
                    if limit is not None and yielded_count >= limit:
                        return
    else:
        raise ValueError(f"Unsupported dataset format: {suffix}. Supported formats: .jsonl, .json, .yaml, .csv")


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
