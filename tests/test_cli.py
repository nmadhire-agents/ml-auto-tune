from pathlib import Path

from ml_auto_tune.cli import main


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
