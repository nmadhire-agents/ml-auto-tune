# ML Auto-Tune Autoresearch Program

You are running bounded, config-only autoresearch for this repository.

## Goal

Improve the configured validation metric while keeping experiments auditable and reproducible.

## Allowed Changes

Autoresearch may only suggest safe YAML config patches for:

- `models`
- `optimization.n_trials`
- `optimization.repeated_splits`
- `optimization.plateau_trials`
- `optimization.min_delta`
- `data.features`
- `advisor.enabled`
- `advisor.provider`
- `advisor.trigger`

## Disallowed Changes

Do not suggest edits to Python source files, dependencies, lockfiles, committed sample data, evaluation metric implementations, output artifact schemas, or shell commands.

## Keep/Discard Rule

The baseline is kept. Later experiments are kept only when they improve the configured metric by at least `research.improvement_min_delta`. Otherwise they are discarded but still logged.

## Output Expectations

Every suggestion should include:

- a short description
- a hypothesis
- a safe `config_patch`
- expected improvement
- risk notes

