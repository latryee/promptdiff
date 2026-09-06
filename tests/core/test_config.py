"""Unit tests for Config and Dataset Loaders."""

from pathlib import Path

from promptdiff.core.config import (
    ProjectConfig,
    load_dataset,
    load_project_config,
    load_prompt_file,
)


def test_load_prompt_file(tmp_path: Path):
    prompt_file = tmp_path / "test.txt"
    prompt_file.write_text("Hello {{name}}", encoding="utf-8")

    pv = load_prompt_file(str(prompt_file), version_name="v1", model="gpt-4o")
    assert pv.name == "v1"
    assert pv.template == "Hello {{name}}"
    assert pv.render({"name": "World"}) == "Hello World"


def test_load_dataset_jsonl(tmp_path: Path):
    jsonl_file = tmp_path / "cases.jsonl"
    jsonl_file.write_text(
        '{"id": "tc1", "vars": {"query": "test 1"}}\n'
        '{"id": "tc2", "description": "desc 2", "vars": {"query": "test 2"}}\n',
        encoding="utf-8",
    )

    cases = load_dataset(str(jsonl_file))
    assert len(cases) == 2
    assert cases[0].id == "tc1"
    assert cases[0].vars["query"] == "test 1"
    assert cases[1].description == "desc 2"


def test_load_dataset_json(tmp_path: Path):
    json_file = tmp_path / "cases.json"
    json_file.write_text(
        '[{"id": "tc1", "query": "test 1"}, {"id": "tc2", "query": "test 2"}]',
        encoding="utf-8",
    )

    cases = load_dataset(str(json_file))
    assert len(cases) == 2
    assert cases[0].id == "tc1"


def test_load_dataset_yaml(tmp_path: Path):
    yaml_file = tmp_path / "cases.yaml"
    yaml_file.write_text(
        "- id: tc_yaml_1\n  vars:\n    query: hello yaml\n",
        encoding="utf-8",
    )

    cases = load_dataset(str(yaml_file))
    assert len(cases) == 1
    assert cases[0].id == "tc_yaml_1"


def test_load_dataset_csv(tmp_path: Path):
    csv_file = tmp_path / "cases.csv"
    csv_file.write_text("id,query\ntc_csv_1,hello csv\n", encoding="utf-8")

    cases = load_dataset(str(csv_file))
    assert len(cases) == 1
    assert cases[0].id == "tc_csv_1"


def test_load_project_config(tmp_path: Path):
    cfg_file = tmp_path / "promptdiff.yaml"
    cfg_file.write_text(
        "model: claude-3-5-sonnet\ntemperature: 0.2\nevaluators: [json_validity, latency]\n",
        encoding="utf-8",
    )

    config = load_project_config(str(cfg_file))
    assert config.model == "claude-3-5-sonnet"
    assert config.temperature == 0.2
    assert "json_validity" in config.evaluators


def test_load_prompt_file_missing_path_logs_warning(caplog) -> None:
    """Ensure missing file-like paths log a clear warning while plain strings do not."""
    import logging

    with caplog.at_level(logging.WARNING):
        # 1. Path-like string with extension
        pv = load_prompt_file("prompts/non_existent_prompt.txt")
        assert pv.template == "prompts/non_existent_prompt.txt"
        assert any(
            "looks like a file path but not found, treating as literal prompt" in r.message for r in caplog.records
        )

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        # 2. Pure literal prompt string without file extensions or path separators
        pv_literal = load_prompt_file("You are a helpful customer support agent.")
        assert pv_literal.template == "You are a helpful customer support agent."
        assert len(caplog.records) == 0


def test_load_dataset_malformed_jsonl(tmp_path: Path) -> None:
    import pytest

    from promptdiff.core.config import DatasetError

    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text('{"id": "tc1", "vars": {}}\n{NOT VALID JSON}\n', encoding="utf-8")

    with pytest.raises(DatasetError) as exc_info:
        load_dataset(str(bad_jsonl))
    assert exc_info.value.line_number == 2
    assert "bad.jsonl" in str(exc_info.value)


def test_load_dataset_filtering_and_limit(tmp_path: Path) -> None:
    jsonl_file = tmp_path / "tagged.jsonl"
    jsonl_file.write_text(
        '{"id": "tc1", "tags": ["smoke", "fast"], "vars": {"x": 1}}\n'
        '{"id": "tc2", "tags": ["slow"], "vars": {"x": 2}}\n'
        '{"id": "tc3", "tags": ["smoke"], "vars": {"x": 3}}\n'
        '{"id": "tc4", "tags": ["smoke"], "vars": {"x": 4}}\n',
        encoding="utf-8",
    )

    smoke_cases = load_dataset(str(jsonl_file), tags=["smoke"])
    assert len(smoke_cases) == 3
    assert [c.id for c in smoke_cases] == ["tc1", "tc3", "tc4"]

    limited_cases = load_dataset(str(jsonl_file), tags=["smoke"], limit=2)
    assert len(limited_cases) == 2
    assert [c.id for c in limited_cases] == ["tc1", "tc3"]


def test_duplicate_id_warning(tmp_path: Path, caplog) -> None:
    import logging

    dup_file = tmp_path / "dup.jsonl"
    dup_file.write_text(
        '{"id": "dup_1", "vars": {"x": 1}}\n{"id": "dup_1", "vars": {"x": 2}}\n',
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        cases = load_dataset(str(dup_file))
        assert len(cases) == 2
        assert any("Duplicate testcase ID 'dup_1'" in r.message for r in caplog.records)


def test_stream_dataset(tmp_path: Path) -> None:
    from promptdiff.core.config import stream_dataset

    jsonl_file = tmp_path / "stream.jsonl"
    jsonl_file.write_text(
        '{"id": "tc1", "vars": {"v": 1}}\n{"id": "tc2", "vars": {"v": 2}}\n',
        encoding="utf-8",
    )

    items = list(stream_dataset(str(jsonl_file)))
    assert len(items) == 2
    assert items[0].id == "tc1"
    assert items[1].id == "tc2"


def test_project_config_extended_fields() -> None:
    cfg = ProjectConfig(
        timeout=30.0,
        cache_ttl=3600,
        tags=["smoke"],
        limit=50,
        redact=True,
        experiment_id="exp-99",
    )
    assert cfg.timeout == 30.0
    assert cfg.cache_ttl == 3600
    assert cfg.tags == ["smoke"]
    assert cfg.limit == 50
    assert cfg.redact is True
    assert cfg.experiment_id == "exp-99"
