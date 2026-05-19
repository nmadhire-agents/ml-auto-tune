# ml-auto-tune

Config-driven local batch tuning for tabular **sklearn regression and classification** with Optuna and optional LLM advice.

The goal is to let users control the workflow from YAML (dataset, target/features, models, optimization, advisor, outputs) while keeping tuning reproducible and auditable.

All Python commands in this repository should be run through `uv`.

## What this project does

- Loads regression or classification CSV data.
- Builds sklearn preprocessing + estimator pipelines.
- Tunes model family and hyperparameters with Optuna (TPE sampler).
- Supports deterministic repeated train/validation splits.
- Captures validation metrics per trial.
  - Regression: `rmse`, `mae`, `r2`
  - Classification: `accuracy`, `f1_macro`, `roc_auc`
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

Run the bundled regression sample workflow:

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

Run the bundled classification sample workflow:

```bash
uv run ml-auto-tune run --config configs/classification_example.yaml
```

Expected output:

```text
Best score: ...
Artifacts: .../runs/classification-example
```

## LLM advisor: what changes?

The optimizer does not require an LLM. Optuna still runs trials, scores models, selects the best trial, and writes artifacts without any advisor.

| Mode | What happens | What you gain | What you do not get |
| --- | --- | --- | --- |
| No LLM | Runs deterministic tuning and writes metrics, trials, model, and summary. | Fast, cheap, reproducible tuning with no credentials or network. | No natural-language interpretation, no suggestions when the search space is too narrow, no diagnosis of why progress stalled. |
| Mock advisor | Uses local deterministic advice for tests and demos. | Exercises the advisor workflow without secrets or API calls. | Advice is generic and not truly reasoned over the problem. |
| OpenAI-compatible advisor | Sends compact trial history, metrics, current best result, model choices, and response schema to an LLM. | Human-readable diagnosis, suggestions for next search-space changes, warnings about weak baselines, and a written explanation in `advisor_advice.md`. | It does not replace validation metrics, and v1 only auto-applies safe structured `model_candidates` suggestions. Free-form advice is recorded for review. |

Use an LLM when you want help interpreting the run, deciding whether the model is likely “good enough,” or choosing what to try next after a plateau. Skip the LLM when you only need repeatable metric-driven tuning.

## Real example: repeated linear regression with advice

The bundled example trains the same simple linear regression model multiple times and asks the advisor after every trial.

Key config:

```yaml
optimization:
  metric: rmse
  n_trials: 5
  repeated_splits: true

advisor:
  enabled: true
  provider: mock
  trigger: each_trial

models:
  - linear_regression
```

Because plain `LinearRegression` has no meaningful hyperparameters, the example uses deterministic repeated splits to produce multiple validation views of the same model. Trial `0` uses split seed `42`, trial `1` uses `43`, and so on. The tuner minimizes RMSE and keeps the best observed validation result.

Run it:

```bash
uv run ml-auto-tune run --config configs/example.yaml
```

Example output:

```text
Best score: 46.427628
Artifacts: /.../runs/example
```

Observed trial metrics from the bundled sample data:

| Trial | Split seed | Model | RMSE | MAE | R2 |
| ---: | ---: | --- | ---: | ---: | ---: |
| 0 | 42 | `linear_regression` | 56.8769 | 47.6459 | 0.4272 |
| 1 | 43 | `linear_regression` | 46.4276 | 38.3666 | 0.6024 |
| 2 | 44 | `linear_regression` | 58.4801 | 45.9821 | 0.2480 |
| 3 | 45 | `linear_regression` | 49.2586 | 40.7508 | 0.5659 |
| 4 | 46 | `linear_regression` | 51.4164 | 40.0569 | 0.5579 |

Best result:

```json
{
  "optimized_metric": "rmse",
  "direction": "minimize",
  "best_score": 46.427628004000084,
  "best_params": {
    "model": "linear_regression"
  },
  "best_split_random_state": 43,
  "validation_metrics": {
    "rmse": 46.427628004000084,
    "mae": 38.36655564277805,
    "r2": 0.6024026519069032
  }
}
```

To ask OpenAI for one advisor response after the repeated linear-regression trials, use the OpenAI-compatible provider and `trigger: end`:

```yaml
advisor:
  enabled: true
  provider: openai_compatible
  trigger: end
```

Set credentials:

```bash
export ML_AUTO_TUNE_LLM_API_KEY="$OPENAI_API_KEY"
export ML_AUTO_TUNE_LLM_MODEL="gpt-4.1-mini"
```

Run the same five-trial linear-regression workflow. The OpenAI advisor receives the best score, the best split seed, all recent trial metrics, and the final validation metrics.

The advisor call uses this system instruction:

```text
You advise an automated sklearn model tuner. Use validation metrics to assess whether the current model appears strong. Return compact JSON with keys markdown and structured_suggestions. Only suggest model_candidates from the provided allowed models.
```

For this run, the user prompt sent to the advisor looked like this:

```json
{
  "study_name": "sample-linear-regression",
  "metric": "rmse",
  "direction": "minimize",
  "best_score": 46.427628004000084,
  "best_params": {
    "model": "linear_regression"
  },
  "trials_since_improvement": 3,
  "allowed_models": [
    "linear_regression"
  ],
  "active_models": [
    "linear_regression"
  ],
  "recent_trials": [
    {
      "number": 0,
      "value": 56.87686219678892,
      "params": {
        "model": "linear_regression"
      },
      "state": "COMPLETE",
      "split_random_state": 42,
      "validation_metrics": {
        "rmse": 56.87686219678892,
        "mae": 47.64585529441712,
        "r2": 0.4271835845269567
      }
    },
    {
      "number": 1,
      "value": 46.427628004000084,
      "params": {
        "model": "linear_regression"
      },
      "state": "COMPLETE",
      "split_random_state": 43,
      "validation_metrics": {
        "rmse": 46.427628004000084,
        "mae": 38.36655564277805,
        "r2": 0.6024026519069032
      }
    },
    {
      "number": 2,
      "value": 58.48014802364387,
      "params": {
        "model": "linear_regression"
      },
      "state": "COMPLETE",
      "split_random_state": 44,
      "validation_metrics": {
        "rmse": 58.48014802364387,
        "mae": 45.98208146689345,
        "r2": 0.24796465445427862
      }
    },
    {
      "number": 3,
      "value": 49.25863864410643,
      "params": {
        "model": "linear_regression"
      },
      "state": "COMPLETE",
      "split_random_state": 45,
      "validation_metrics": {
        "rmse": 49.25863864410643,
        "mae": 40.750769154611916,
        "r2": 0.5658526187837658
      }
    },
    {
      "number": 4,
      "value": 51.416383959104,
      "params": {
        "model": "linear_regression"
      },
      "state": "COMPLETE",
      "split_random_state": 46,
      "validation_metrics": {
        "rmse": 51.416383959104,
        "mae": 40.05693552923261,
        "r2": 0.5578709979888179
      }
    }
  ],
  "validation_metrics": {
    "rmse": 46.427628004000084,
    "mae": 38.36655564277805,
    "r2": 0.6024026519069032
  },
  "response_schema": {
    "markdown": "human-readable tuning advice",
    "structured_suggestions": {
      "model_candidates": "optional list containing only allowed model names"
    }
  }
}
```

The real OpenAI response for that prompt was:

```text
The current linear regression model achieves an RMSE of 46.43 and an R² of 0.60, indicating moderate predictive performance. Given the limited model choice (only linear regression allowed) and recent trials showing no improvement beyond this RMSE, the model appears reasonably tuned within these constraints. Consider expanding allowed models or feature engineering to improve performance further.
```

The system records the advice in `advisor_advice.md`. It only auto-applies safe structured suggestions that map to configured model candidates; free-form guidance is stored for review.

## Real example: classification tuning

The classification example uses `data/sample_classification.csv`, derived from sklearn's breast-cancer dataset. It tunes several classifier families and optimizes `f1_macro`.

Key config:

```yaml
task: classification

optimization:
  metric: f1_macro
  n_trials: 8
  repeated_splits: true

advisor:
  enabled: true
  provider: mock
  trigger: end

models:
  - logistic_regression
  - random_forest_classifier
  - hist_gradient_boosting_classifier
```

Run it:

```bash
uv run ml-auto-tune run --config configs/classification_example.yaml
```

Example output:

```text
Best score: 0.975396
Artifacts: /.../runs/classification-example
```

Observed trial metrics from the bundled sample data:

| Trial | Split seed | Model | Accuracy | F1 macro | ROC-AUC |
| ---: | ---: | --- | ---: | ---: | ---: |
| 0 | 42 | `random_forest_classifier` | 0.9333 | 0.9237 | 0.9828 |
| 1 | 43 | `logistic_regression` | 0.8444 | 0.8062 | 0.9978 |
| 2 | 44 | `logistic_regression` | 0.9556 | 0.9500 | 1.0000 |
| 3 | 45 | `hist_gradient_boosting_classifier` | 0.9111 | 0.9055 | 0.9892 |
| 4 | 46 | `hist_gradient_boosting_classifier` | 0.9556 | 0.9500 | 0.9978 |
| 5 | 47 | `random_forest_classifier` | 0.9778 | 0.9754 | 0.9978 |
| 6 | 48 | `hist_gradient_boosting_classifier` | 0.9778 | 0.9754 | 1.0000 |
| 7 | 49 | `logistic_regression` | 0.9333 | 0.9237 | 1.0000 |

Best result:

```json
{
  "optimized_metric": "f1_macro",
  "direction": "maximize",
  "best_score": 0.9753963914707491,
  "best_params": {
    "model": "random_forest_classifier",
    "rf_classifier_n_estimators": 50,
    "rf_classifier_max_depth": 18,
    "rf_classifier_min_samples_leaf": 8,
    "rf_classifier_max_features": 0.8850384088698767
  },
  "best_split_random_state": 47,
  "validation_metrics": {
    "accuracy": 0.9777777777777777,
    "f1_macro": 0.9753963914707491,
    "roc_auc": 0.9978448275862069
  }
}
```

The advisor note for this run is stored in `runs/classification-example/advisor_advice.md`. With the mock advisor, it is intentionally conservative:

```text
Mock advisor: evaluated the current validation metrics. Accuracy=0.9778, F1 macro=0.9754, ROC-AUC=0.9978. Confirm class balance and compare a linear classifier against tree-based classifiers before declaring the model best.
```

## CLI

### Run tuning

```bash
uv run ml-auto-tune run --config configs/example.yaml
uv run ml-auto-tune run --config configs/classification_example.yaml
```

### Generate sample data

```bash
uv run ml-auto-tune make-sample-data --output data/sample_regression.csv --rows 180
uv run ml-auto-tune make-sample-data --task classification --output data/sample_classification.csv --rows 180
```

Sample-data options:

- `--output`: CSV output path
- `--task`: `regression` or `classification`
- `--rows`: number of rows
- `--random-state`: deterministic sampling seed

## Configuration (YAML-first)

Primary interface:

```text
configs/example.yaml
configs/classification_example.yaml
```

### `task`

- `task: regression`
- `task: classification`

### `data`

- `data.path`: CSV path
- `data.target`: target column name
- `data.features`: optional explicit feature list; defaults to all non-target columns
- `data.validation_size`: validation split fraction
- `optimization.metric`: regression uses `rmse`, `mae`, or `r2`; classification uses `accuracy`, `f1_macro`, or `roc_auc`
- `optimization.n_trials`: Optuna trial budget
- `optimization.plateau_trials`: number of non-improving trials before advisor advice
- `optimization.min_delta`: minimum score change counted as improvement
- `optimization.repeated_splits`: when true, each trial uses a different deterministic train/validation split seed
- `advisor.enabled`: enable or disable advisor calls
- `advisor.provider`: `mock` or `openai_compatible`
- `advisor.trigger`: `plateau`, `end`, or `each_trial`
- `output.directory`: local artifact directory
- `models`: candidate sklearn estimator families

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

Regression:

- `linear_regression`
- `random_forest`
- `extra_trees`
- `hist_gradient_boosting`
- `ridge`
- `elastic_net`

Classification:

- `logistic_regression`
- `random_forest_classifier`
- `extra_trees_classifier`
- `hist_gradient_boosting_classifier`

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

Run the sample workflows:

```bash
uv run ml-auto-tune run --config configs/example.yaml
uv run ml-auto-tune run --config configs/classification_example.yaml
```

Inspect outputs:

```bash
cat runs/example/metrics.json
cat runs/example/advisor_advice.md
cat runs/example/run_summary.md
cat runs/classification-example/metrics.json
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

v1 focuses on local batch tabular sklearn tuning. It does not include:

- a web service/UI
- job scheduler/orchestrator
- experiment database backend
- distributed training
