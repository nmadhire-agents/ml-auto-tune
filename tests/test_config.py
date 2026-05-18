from pathlib import Path

import pytest

from ml_auto_tune.config import parse_config


def test_parse_config_resolves_paths(tmp_path: Path) -> None:
    config = parse_config(
        {
            "data": {"path": "sample.csv", "target": "target"},
            "output": {"directory": "runs/test"},
            "models": ["ridge"],
        },
        base_dir=tmp_path,
    )

    assert config.data.path == tmp_path / "sample.csv"
    assert config.output.directory == tmp_path / "runs/test"
    assert config.direction == "minimize"


def test_parse_config_rejects_unknown_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        parse_config(
            {
                "data": {"path": "sample.csv", "target": "target"},
                "models": ["made_up_model"],
            },
            base_dir=tmp_path,
        )

