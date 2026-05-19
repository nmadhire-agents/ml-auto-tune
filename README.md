# ml-auto-tune

Config-driven local batch tuning for tabular **sklearn regression** with Optuna and optional LLM advice.

The goal is to let users control the workflow from YAML (dataset, target/features, models, optimization, advisor, outputs) while keeping tuning reproducible and auditable.

All Python commands in this repository should be run through `uv`.

## What this project does

- Loads regression CSV data.
- Builds sklearn preprocessing + regressor pipelines.
- Tunes model family and hyperparameters with Optuna (TPE sampler).
- Supports deterministic repeated train/validation splits.
- Captures validation metrics (`rmse`, `mae`, `r2`) per trial.
- Optionally asks an advisor (`mock` or OpenAI-compatible chat completions).
- Applies only safe structured advisor suggestions (`model_candidates` limited to configured models).
- Writes reproducible run artifacts (resolved config, metrics, trial history, model, advisor notes, summary).

## Architecture (simple)

```text
                ┌──────────────────────────────┐
                │       YAML Config File       │
                │   (data, models, optimizer,  │
                │    advisor, output, metric)  │
                └──────────────┬───────────────┘
                               │
                               ▼
                     ┌──────────────────┐
                     │ CLI / Python API │
                     │ load_config(...) │
                     └────────┬─────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  Tuning Orchestrator│
                    │   run_tuning(...)   │
                    └───────┬─────────────┘
                            │
      ┌─────────────────────┼─────────────────────┐
      ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐    ┌───────────────────┐
│ Data Loader   │   │ Model Builder │    │ Advisor (optional)│
│ + Split Logic │   │ sklearn pipe  │    │ mock / openai comp│
└──────┬────────┘   └──────┬────────┘    └─────────┬─────────┘
       │                   │                        │
       └──────────────┬────┴──────────────┬─────────┘
                      ▼                   ▼
                ┌────────────────────────────────────┐
                │ Optuna Study (trials + best trial)│
                └────────────────┬───────────────────┘
                                 ▼
                   ┌─────────────────────────────┐
                   │ Artifact Writer (runs/...)  │
                   │ config, metrics, trials,    │
                   │ model, advisor_advice,      │
                   │ run_summary                 │
                   └─────────────────────────────┘
```

## Requirements

- `uv`
- Python 3.12 (pinned by `.python-version`)

Install dependencies:

```bash
uv sync
```

## Quick start

Run the bundled sample workflow:

```bash
uv run ml-auto-tune run --config configs/example.yaml
```

Expected output:

```text
Best score: ...
Artifacts: .../runs/example
```

The sample config is intentionally local/offline-friendly:

- it tunes `linear_regression`
- uses repeated deterministic validation splits
- calls the **mock advisor** on `each_trial`
- needs no network and no LLM credentials

## CLI

### Run tuning

```bash
uv run ml-auto-tune run --config configs/example.yaml
```

### Generate sample data

```bash
uv run ml-auto-tune make-sample-data --output data/sample_regression.csv --rows 180
```

Sample-data options:

- `--output`: CSV output path
- `--rows`: number of rows
- `--random-state`: deterministic sampling seed

## Configuration (YAML-first)

Primary interface:

```text
configs/example.yaml
```

### `data`

- `data.path`: CSV path
- `data.target`: target column name
- `data.features`: optional explicit feature list; defaults to all non-target columns
- `data.validation_size`: validation split fraction
- `optimization.metric`: one of `rmse`, `mae`, or `r2`
- `optimization.n_trials`: Optuna trial budget
- `optimization.plateau_trials`: number of non-improving trials before advisor advice
- `optimization.min_delta`: minimum score change counted as improvement
- `optimization.repeated_splits`: when true, each trial uses a different deterministic train/validation split seed
- `advisor.enabled`: enable or disable advisor calls
- `advisor.provider`: `mock` or `openai_compatible`
- `advisor.trigger`: `plateau`, `end`, or `each_trial`
- `output.directory`: local artifact directory
- `models`: candidate sklearn regressor families

### `optimization`

- `metric` (default `rmse`): one of `rmse`, `mae`, `r2`
- `n_trials` (default `20`): Optuna trial count
- `timeout_seconds` (optional): per-optimize timeout
- `plateau_trials` (default `5`): non-improving trials before plateau action
- `min_delta` (default `0.001`): minimum change to count as improvement
- `study_name` (default `ml-auto-tune`)
- `random_state` (default `42`): TPE seed
- `repeated_splits` (default `false`): if `true`, uses `data.random_state + trial_number`

### `advisor`

- `enabled` (default `true`)
- `provider`: `mock` or `openai_compatible`
- `trigger`: `plateau`, `end`, or `each_trial`
- optional explicit credentials fields in config:
  - `api_key`
  - `base_url`
  - `model`

### `output`

- `directory`: local artifact directory (e.g. `../runs/example`)

### `models`

Candidate model families:

- `linear_regression`
- `random_forest`
- `extra_trees`
- `hist_gradient_boosting`
- `ridge`
- `elastic_net`

## Advisor behavior and safety

The advisor flow is conservative and auditable:

- Every advisor response is written to `advisor_advice.md`.
- Validation metrics (`RMSE`, `MAE`, `R2`) are included in advisor context when available.
- Structured suggestions are parsed, but only known safe controls are applied.
- In v1, safe auto-applied suggestions are limited to `model_candidates` that match configured model names.
- Non-safe or non-structured guidance is recorded but not blindly executed.

For OpenAI-compatible usage, set environment variables:

```bash
export ML_AUTO_TUNE_LLM_API_KEY="..."
export ML_AUTO_TUNE_LLM_BASE_URL="https://api.openai.com/v1"
export ML_AUTO_TUNE_LLM_MODEL="..."
```

## Artifacts

Each run writes under `output.directory` (example: `runs/example/`):

- `config.resolved.yaml`
- `metrics.json`
- `trials.csv`
- `best_model.joblib`
- `advisor_advice.md`
- `run_summary.md`

`runs/` is git-ignored as disposable local output.

## Python API

```python
from ml_auto_tune import load_config, run_tuning

config = load_config("configs/example.yaml")
result = run_tuning(config)
print(result.best_score)
print(result.metrics)
```

Advisor extension points:

```python
from ml_auto_tune import Advisor, AdvisorContext, AdvisorResponse

def advise(self, context: AdvisorContext) -> AdvisorResponse:
    ...
```

## Development

Run tests:

```bash
uv run pytest
```

Run the sample workflow:

```bash
uv run ml-auto-tune run --config configs/example.yaml
```

Inspect outputs:

```bash
cat runs/example/metrics.json
cat runs/example/advisor_advice.md
cat runs/example/run_summary.md
```

Useful files:

- `src/ml_auto_tune/config.py`: YAML schema parsing/validation
- `src/ml_auto_tune/data.py`: CSV loading + split logic
- `src/ml_auto_tune/models.py`: preprocessing + model search spaces
- `src/ml_auto_tune/tuning.py`: Optuna loop + artifact writing
- `src/ml_auto_tune/advisor.py`: mock and OpenAI-compatible advisors
- `src/ml_auto_tune/sample_data.py`: sample regression data generator
- `tests/`: unit and smoke tests

## Current scope

v1 focuses on local batch **regression** tuning. It does not include:

- a web service/UI
- job scheduler/orchestrator
- experiment database backend
- distributed training
- classification workflows
