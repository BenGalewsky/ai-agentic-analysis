"""Load prompts from the MLflow Prompt Registry and render them."""

from __future__ import annotations

import mlflow


def resolve_prompt(name: str, version: str | int | None):
    """Load a prompt version; ``None`` means the highest registered version."""
    if version is None:
        versions = [int(v.version) for v in mlflow.MlflowClient().search_prompt_versions(name)]
        if not versions:
            raise SystemExit(f"No versions registered for prompt {name!r}")
        version = max(versions)
    return mlflow.genai.load_prompt(f"prompts:/{name}/{version}")


def render_prompt(prompt, variables: dict[str, str]) -> str:
    if not prompt.variables:
        return prompt.template
    missing = set(prompt.variables) - set(variables)
    if missing:
        raise SystemExit(f"Prompt {prompt.name!r} needs --var for: {', '.join(sorted(missing))}")
    return prompt.format(**variables)
