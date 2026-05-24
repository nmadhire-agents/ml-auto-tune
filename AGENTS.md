# Repository Instructions

## Project Shape

This is a `uv`-managed Python 3.12 project for local batch tuning of tabular sklearn regression and classification models.

Use `uv` for every Python command. Do not rely on bare `python`, because local system Python may not match the project runtime.

## Common Commands

Install or sync dependencies:

```bash
uv sync
```

Run tests:

```bash
uv run pytest
```

Run the sample workflows:

```bash
uv run ml-auto-tune run --config configs/example.yaml
uv run ml-auto-tune run --config configs/classification_example.yaml
uv run ml-auto-tune autoresearch --config configs/research_example.yaml
```

Regenerate sample data:

```bash
uv run ml-auto-tune make-sample-data --output data/sample_regression.csv --rows 180
uv run ml-auto-tune make-sample-data --task classification --output data/sample_classification.csv --rows 180
```

## Implementation Notes

- Keep the public CLI entrypoint as `ml-auto-tune`.
- Keep the primary config format as YAML.
- Keep CSV tabular data as the v1 input shape.
- Keep no-network tests possible by using the mock advisor.
- The example config intentionally trains `linear_regression` across repeated deterministic validation splits and asks the advisor on `each_trial`.
- The classification example uses sklearn breast-cancer data and tunes classifier families with `f1_macro`.
- Autoresearch is config-only: it may generate experiment configs and run them, but it must not edit Python source, dependencies, lockfiles, committed sample data, or evaluation code.
- Treat `runs/` as disposable local output; it is ignored by git.
- Do not commit secrets, `.env`, local virtual environments, or generated run artifacts.

## Advisor Behavior

The LLM advisor must be auditable and conservative:

- write advice to artifacts
- allow mock advisor execution without credentials
- use OpenAI-compatible chat completions only when explicitly configured
- read credentials from `ML_AUTO_TUNE_LLM_API_KEY`, `ML_AUTO_TUNE_LLM_BASE_URL`, and `ML_AUTO_TUNE_LLM_MODEL`
- only apply structured suggestions that map to known safe controls
- include validation metrics when available: RMSE/MAE/R2 for regression and accuracy/F1/ROC-AUC for classification

For v1, safe applied suggestions are model-candidate hints that match configured model names. Other advice should be recorded, not blindly executed.

## Testing Expectations

When changing tuning, config, advisor, sample-data, or CLI behavior, run:

```bash
uv run pytest
```

For user-facing workflow changes, also run:

```bash
uv run ml-auto-tune run --config configs/example.yaml
uv run ml-auto-tune run --config configs/classification_example.yaml
uv run ml-auto-tune autoresearch --config configs/research_example.yaml
```

The example run should produce:

- `config.resolved.yaml`
- `metrics.json`
- `trials.csv`
- `best_model.joblib`
- `advisor_advice.md`
- `run_summary.md`

The autoresearch example should additionally produce:

- `research_results.tsv`
- `research_log.md`
- `best_config.yaml`
- `best_metrics.json`
- `llm_suggestions.jsonl`
