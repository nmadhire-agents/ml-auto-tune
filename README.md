# ml-auto-tune

`ml-auto-tune` is a small Python tool for trying different machine learning models and keeping track of which one works best.

It is designed for tabular CSV data, such as spreadsheets exported to `.csv`.

You can use it for:

- **Regression**: predict a number, such as price, demand, risk score, or duration.
- **Classification**: predict a category, such as yes/no, fraud/not fraud, or class A/B/C.
- **LLM advice**: ask an LLM to explain results and suggest what to try next.
- **Autoresearch**: run a bounded loop of safe configuration-only experiments.

All Python commands in this repo should be run with `uv`.

## Quick Start

Install dependencies:

```bash
uv sync
```

Run the regression example:

```bash
uv run ml-auto-tune run --config configs/example.yaml
```

Run the classification example:

```bash
uv run ml-auto-tune run --config configs/classification_example.yaml
```

Run the autoresearch example:

```bash
uv run ml-auto-tune autoresearch --config configs/research_example.yaml
```

Each command prints a best score and writes output files under `runs/`.

## What The Tool Does

At a high level:

1. Reads your CSV file.
2. Splits the data into training and validation sets.
3. Tries one or more sklearn models.
4. Measures how well each model performs.
5. Saves the best model and all run details.
6. Optionally asks an LLM for advice.

```text
CSV data + YAML config
        |
        v
Load data -> split train/validation -> try sklearn models -> score metrics
        |                                                   |
        v                                                   v
 saved artifacts                                    optional LLM advice
```

## Important ML Terms

If you are new to ML, these are the key ideas used in this repo.

| Term | Meaning |
| --- | --- |
| Dataset | Your CSV file. Rows are examples, columns are values. |
| Feature | An input column used to make predictions. |
| Target | The column you want to predict. |
| Model | The algorithm used to learn from data. |
| Training set | Data used to fit the model. |
| Validation set | Data held aside to estimate how well the model works. |
| Metric | A number that scores model quality. |
| Tuning | Trying model/settings combinations to improve the metric. |
| Artifact | A saved output file, such as metrics, trial history, or model file. |

## Regression vs Classification

Use **regression** when the answer is a number.

Examples:

- predict house price
- predict sales next week
- predict delivery time

Regression metrics:

| Metric | Simple meaning | Better value |
| --- | --- | --- |
| `rmse` | Average prediction error, with larger mistakes penalized more. | Lower |
| `mae` | Average absolute prediction error. | Lower |
| `r2` | How much variation the model explains. | Higher |

Use **classification** when the answer is a class or label.

Examples:

- predict whether a customer will churn
- predict whether a transaction is fraud
- predict whether a tumor is benign or malignant

Classification metrics:

| Metric | Simple meaning | Better value |
| --- | --- | --- |
| `accuracy` | Fraction of predictions that are correct. | Higher |
| `f1_macro` | Balanced score across classes, useful when classes are uneven. | Higher |
| `roc_auc` | How well the model separates classes. | Higher |

## Example 1: Regression

The bundled regression example uses `data/sample_regression.csv`, based on sklearn's diabetes dataset.

Run:

```bash
uv run ml-auto-tune run --config configs/example.yaml
```

Example result:

```text
Best score: 46.427628
Artifacts: .../runs/example
```

This example trains `linear_regression` multiple times using different deterministic train/validation splits.

Example best metrics:

```json
{
  "optimized_metric": "rmse",
  "direction": "minimize",
  "best_score": 46.427628004000084,
  "best_params": {
    "model": "linear_regression"
  },
  "validation_metrics": {
    "rmse": 46.427628004000084,
    "mae": 38.36655564277805,
    "r2": 0.6024026519069032
  }
}
```

How to read this:

- The tool optimized `rmse`.
- Because RMSE is an error metric, lower is better.
- The best validation RMSE was about `46.43`.
- The model explained some signal in the data, with `r2` around `0.60`.

## Example 2: Classification

The bundled classification example uses `data/sample_classification.csv`, based on sklearn's breast-cancer dataset.

Run:

```bash
uv run ml-auto-tune run --config configs/classification_example.yaml
```

Example result:

```text
Best score: 0.975396
Artifacts: .../runs/classification-example
```

Example best metrics:

```json
{
  "optimized_metric": "f1_macro",
  "direction": "maximize",
  "best_score": 0.9753963914707491,
  "best_params": {
    "model": "random_forest_classifier"
  },
  "validation_metrics": {
    "accuracy": 0.9777777777777777,
    "f1_macro": 0.9753963914707491,
    "roc_auc": 0.9978448275862069
  }
}
```

How to read this:

- The tool optimized `f1_macro`.
- Because F1 is a quality score, higher is better.
- The best validation F1 macro was about `0.975`.
- Accuracy and ROC-AUC were also high.

## Example 3: Autoresearch

Autoresearch runs several safe experiments automatically.

It does not edit Python source code. It only creates temporary YAML experiment configs, runs them, and compares scores.

Run:

```bash
uv run ml-auto-tune autoresearch --config configs/research_example.yaml
```

Example result:

```text
Best score: 45.175201
Best experiment: 003
Artifacts: .../runs/autoresearch-example
```

Autoresearch writes:

- `research_results.tsv`: one row per experiment
- `research_log.md`: human-readable notes
- `configs/experiment_000.yaml`, `configs/experiment_001.yaml`, etc.
- `best_config.yaml`: the best generated config
- `best_metrics.json`: metrics for the best experiment
- `llm_suggestions.jsonl`: suggestions used during the loop

The loop is:

```text
baseline config
     |
     v
run experiment -> score it -> keep if better, discard if worse
     |
     v
suggest next safe config change
     |
     v
stop after max_experiments
```

## LLM Advice: What You Gain

The tool works without an LLM. The LLM is optional.

| Mode | What happens | Good for |
| --- | --- | --- |
| No LLM | Runs tuning and saves metrics. | Fast, reproducible experiments. |
| Mock advisor | Writes local example advice without network calls. | Tests and demos. |
| OpenAI-compatible advisor | Sends run summary and metrics to an LLM. | Human-readable explanation and next-step suggestions. |

What an LLM can help with:

- explain whether a score looks strong or weak
- notice when the search space is too narrow
- suggest model families to try next
- summarize why tuning may be stuck
- write advice to `advisor_advice.md`

What an LLM does not replace:

- validation metrics
- domain knowledge
- careful data cleaning
- proper train/test evaluation

For OpenAI-compatible advisor usage:

```bash
export ML_AUTO_TUNE_LLM_API_KEY="..."
export ML_AUTO_TUNE_LLM_BASE_URL="https://api.openai.com/v1"
export ML_AUTO_TUNE_LLM_MODEL="gpt-4.1-mini"
```

Then set the config advisor provider:

```yaml
advisor:
  enabled: true
  provider: openai_compatible
  trigger: end
```

## Project Architecture

```text
                   YAML config
                       |
                       v
                CLI / Python API
                       |
        +--------------+---------------+
        |                              |
        v                              v
   one run                         autoresearch
 run_tuning()                   run_autoresearch()
        |                              |
        |                    generate safe config patches
        |                              |
        +--------------+---------------+
                       |
                       v
              shared tuning engine
                       |
        +--------------+---------------+
        |              |               |
        v              v               v
   data loader     model builder    advisor
   CSV split       sklearn pipe     mock/OpenAI
        |              |               |
        +--------------+---------------+
                       |
                       v
             artifacts written to runs/
```

## Configuration

The tool is controlled by YAML files.

Bundled examples:

```text
configs/example.yaml
configs/classification_example.yaml
configs/research_example.yaml
```

Generic templates for your own data:

```text
configs/templates/regression.yaml
configs/templates/classification.yaml
```

Relative paths in config files are resolved relative to the config file location. Absolute paths also work.

### Minimal Regression Config

```yaml
task: regression

data:
  path: /path/to/your_data.csv
  target: target_column_name
  validation_size: 0.25

optimization:
  metric: rmse
  n_trials: 30

advisor:
  enabled: false

output:
  directory: runs/my-regression-run

models:
  - linear_regression
  - ridge
  - random_forest
```

### Minimal Classification Config

```yaml
task: classification

data:
  path: /path/to/your_data.csv
  target: target_column_name
  validation_size: 0.25

optimization:
  metric: f1_macro
  n_trials: 30

advisor:
  enabled: false

output:
  directory: runs/my-classification-run

models:
  - logistic_regression
  - random_forest_classifier
```

### Important Config Fields

| Field | Meaning |
| --- | --- |
| `task` | `regression` or `classification`. |
| `data.path` | Path to your CSV file. |
| `data.target` | Column to predict. |
| `data.features` | Optional list of input columns. If omitted, all non-target columns are used. |
| `data.validation_size` | Fraction of data used for validation, such as `0.25`. |
| `optimization.metric` | Metric to optimize. |
| `optimization.n_trials` | Number of model/settings attempts. |
| `advisor.enabled` | Whether to ask for advice. |
| `research.enabled` | Whether this config can be used with `autoresearch`. |
| `output.directory` | Where artifacts are saved. |
| `models` | Model families to try. |

## Supported Models

Regression models:

- `linear_regression`
- `ridge`
- `elastic_net`
- `random_forest`
- `extra_trees`
- `hist_gradient_boosting`

Classification models:

- `logistic_regression`
- `random_forest_classifier`
- `extra_trees_classifier`
- `hist_gradient_boosting_classifier`

## Output Files

Normal tuning writes:

- `config.resolved.yaml`: the final config used
- `metrics.json`: best score and validation metrics
- `trials.csv`: all model attempts
- `best_model.joblib`: saved sklearn model pipeline
- `advisor_advice.md`: LLM or mock advice, if enabled
- `run_summary.md`: short summary

Autoresearch also writes:

- `research_results.tsv`
- `research_log.md`
- `best_config.yaml`
- `best_metrics.json`
- `llm_suggestions.jsonl`

`runs/` is ignored by git because these are local experiment outputs.

## CLI Reference

Run a tuning config:

```bash
uv run ml-auto-tune run --config configs/example.yaml
```

Run autoresearch:

```bash
uv run ml-auto-tune autoresearch --config configs/research_example.yaml
```

Generate sample data:

```bash
uv run ml-auto-tune make-sample-data --output data/sample_regression.csv --rows 180
uv run ml-auto-tune make-sample-data --task classification --output data/sample_classification.csv --rows 180
```

## Python API

```python
from ml_auto_tune import load_config, run_autoresearch, run_tuning

config = load_config("configs/example.yaml")
result = run_tuning(config)
print(result.best_score)
print(result.metrics)

research_config = load_config("configs/research_example.yaml")
research_result = run_autoresearch(research_config)
print(research_result.best_experiment.score)
```

## Development

Run tests:

```bash
uv run pytest
```

Run all sample workflows:

```bash
uv run ml-auto-tune run --config configs/example.yaml
uv run ml-auto-tune run --config configs/classification_example.yaml
uv run ml-auto-tune autoresearch --config configs/research_example.yaml
```

Useful files:

- `src/ml_auto_tune/config.py`: YAML parsing and validation
- `src/ml_auto_tune/data.py`: CSV loading and train/validation splitting
- `src/ml_auto_tune/models.py`: sklearn model definitions
- `src/ml_auto_tune/tuning.py`: Optuna tuning loop
- `src/ml_auto_tune/advisor.py`: mock and OpenAI-compatible advisors
- `src/ml_auto_tune/autoresearch.py`: bounded config-only research loop
- `tests/`: unit and smoke tests

## Current Scope

This is a local batch tool for tabular sklearn experiments. It does not include:

- a web UI
- a production scheduler
- distributed training
- an experiment database server
