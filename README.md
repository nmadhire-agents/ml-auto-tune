# ml-auto-tune

Auto-tune tabular sklearn regression models with Optuna and optional LLM advice.

This project provides a local-first batch workflow for regression experiments:

- load a CSV dataset
- build an sklearn preprocessing and regression pipeline
- search model and hyperparameter choices with Optuna TPE
- detect tuning plateaus
- ask an LLM advisor for tuning suggestions when configured
- write reproducible run artifacts, metrics, trial history, and the fitted best model

All Python commands in this repository should be run through `uv`.

## Requirements

- `uv`
- Python 3.12, pinned by `.python-version`

Install dependencies:

```bash
uv sync
```

## Quick Start

Run the checked-in sample workflow:

```bash
uv run ml-auto-tune run --config configs/example.yaml
```

Expected output:

```text
Best score: ...
Artifacts: .../runs/example
```

The example uses the mock advisor, so it does not require network access or LLM credentials.

## Sample Data

The repository includes:

```text
data/sample_regression.csv
```

This is a small CSV derived from the built-in sklearn diabetes regression dataset. It includes numeric feature columns, a derived categorical `bmi_band` feature, and the regression target column named `target`.

Regenerate it with:

```bash
uv run ml-auto-tune make-sample-data --output data/sample_regression.csv --rows 180
```

## CLI

Run model tuning:

```bash
uv run ml-auto-tune run --config configs/example.yaml
```

Generate sample data:

```bash
uv run ml-auto-tune make-sample-data --output data/sample_regression.csv --rows 180
```

Available sample-data options:

- `--output`: CSV output path
- `--rows`: number of rows to include
- `--random-state`: deterministic sampling seed

## Configuration

The primary interface is a YAML config file. See:

```text
configs/example.yaml
```

Important sections:

- `data.path`: CSV path
- `data.target`: target column name
- `data.validation_size`: validation split fraction
- `optimization.metric`: one of `rmse`, `mae`, or `r2`
- `optimization.n_trials`: Optuna trial budget
- `optimization.plateau_trials`: number of non-improving trials before advisor advice
- `optimization.min_delta`: minimum score change counted as improvement
- `advisor.enabled`: enable or disable advisor calls
- `advisor.provider`: `mock` or `openai_compatible`
- `advisor.trigger`: `plateau` or `end`
- `output.directory`: local artifact directory
- `models`: candidate sklearn regressor families

Supported model names:

- `random_forest`
- `extra_trees`
- `hist_gradient_boosting`
- `ridge`
- `elastic_net`

## Artifacts

Each run writes artifacts under the configured output directory.

For the example config, artifacts are written to:

```text
runs/example/
```

Generated files:

- `config.resolved.yaml`: normalized config used by the run
- `metrics.json`: optimized metric, best score, best params, and validation metrics
- `trials.csv`: Optuna trial history
- `best_model.joblib`: fitted sklearn pipeline using the best trial parameters
- `advisor_advice.md`: mock or LLM advisor output
- `run_summary.md`: concise human-readable run summary

`runs/` is ignored by git because these files are local experiment outputs.

## LLM Advisor

The advisor is optional. The default example uses:

```yaml
advisor:
  enabled: true
  provider: mock
  trigger: plateau
```

The mock advisor is deterministic and intended for local development, tests, and demos.

To use an OpenAI-compatible chat completions endpoint:

```yaml
advisor:
  enabled: true
  provider: openai_compatible
  trigger: plateau
```

Set these environment variables:

```bash
export ML_AUTO_TUNE_LLM_API_KEY="..."
export ML_AUTO_TUNE_LLM_BASE_URL="https://api.openai.com/v1"
export ML_AUTO_TUNE_LLM_MODEL="..."
```

Advisor behavior:

- receives a compact summary of recent trials, current best score, best params, metric direction, available models, and plateau state
- writes all advice to `advisor_advice.md`
- accepts only safe structured `model_candidates` suggestions that match configured model names
- applies safe model suggestions by enqueueing compatible Optuna trials

## Python API

The package exports:

```python
from ml_auto_tune import TuningConfig, load_config, run_tuning

config = load_config("configs/example.yaml")
result = run_tuning(config)
print(result.best_score)
```

Advisor extension points:

```python
from ml_auto_tune import Advisor, AdvisorContext, AdvisorResponse
```

Custom advisors should implement:

```python
def advise(self, context: AdvisorContext) -> AdvisorResponse:
    ...
```

## Development

Run tests:

```bash
uv run pytest
```

Run the example workflow:

```bash
uv run ml-auto-tune run --config configs/example.yaml
```

Useful files:

- `src/ml_auto_tune/config.py`: config dataclasses and YAML parsing
- `src/ml_auto_tune/models.py`: sklearn preprocessing and model search spaces
- `src/ml_auto_tune/tuning.py`: Optuna orchestration and artifact writing
- `src/ml_auto_tune/advisor.py`: mock and OpenAI-compatible advisors
- `src/ml_auto_tune/sample_data.py`: sklearn-derived sample data generation
- `tests/`: unit and smoke tests

## Current Scope

This is a v1 local batch workflow. It does not include a scheduler, web service, experiment database, distributed training, or classification support.
