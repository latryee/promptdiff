"""Unit tests for Config and Dataset Loaders."""

from pathlib import Path

from promptdiff.core.config import (
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
        "- id: tc_yaml_1\n"
        "  vars:\n"
        "    query: hello yaml\n",
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
        "model: claude-3-5-sonnet\n"
        "temperature: 0.2\n"
        "evaluators: [json_validity, latency]\n",
        encoding="utf-8",
    )

    config = load_project_config(str(cfg_file))
    assert config.model == "claude-3-5-sonnet"
    assert config.temperature == 0.2
    assert "json_validity" in config.evaluators
