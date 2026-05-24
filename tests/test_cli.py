from pathlib import Path

from ml_auto_tune.cli import main
from ml_auto_tune.sample_data import make_sample_data


def test_cli_make_sample_data(tmp_path: Path, capsys) -> None:
    output = tmp_path / "sample.csv"

    main(["make-sample-data", "--output", str(output), "--rows", "30"])

    captured = capsys.readouterr()
    assert output.exists()
    assert "Wrote sample data" in captured.out


def test_cli_make_classification_sample_data(tmp_path: Path) -> None:
    output = tmp_path / "classification.csv"

    main(["make-sample-data", "--task", "classification", "--output", str(output), "--rows", "30"])

    assert output.exists()


def test_cli_autoresearch_smoke(tmp_path: Path) -> None:
    data_path = make_sample_data(tmp_path / "sample.csv", rows=50)
    program_path = tmp_path / "program.md"
    program_path.write_text("Run safe config-only experiments.", encoding="utf-8")
    config_path = tmp_path / "research.yaml"
    config_path.write_text(
        f"""
data:
  path: {data_path}
  target: target
optimization:
  metric: rmse
  n_trials: 1
  study_name: cli-autoresearch-test
advisor:
  enabled: false
research:
  enabled: true
  max_experiments: 1
  llm_enabled: false
  program_path: {program_path}
output:
  directory: {tmp_path / "research-run"}
models:
  - linear_regression
""",
        encoding="utf-8",
    )

    main(["autoresearch", "--config", str(config_path)])

    assert (tmp_path / "research-run" / "research_results.tsv").exists()
