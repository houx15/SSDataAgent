from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YAML = REPO_ROOT / "config" / "llm.yaml"

# Load .env once at import so that any code path reading env vars
# (project_root, data_root, results_root, llm config, ...) sees the values.
# `override=False` means real shell env wins over .env, matching dotenv's
# usual semantics. Only loads if .env exists; tests that don't want this
# can monkey-patch os.environ after import.
_ENV_PATH = REPO_ROOT / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH, override=False)


def project_root() -> Path:
    """Where `real_data/` and `results/` live.

    Defaults to the repo root. Set `SSDA_ROOT=/mnt/disk2/ssda` to relocate
    user-managed data + outputs onto a mounted disk on a cloud box; that
    directory should contain `real_data/` (you upload it) and `results/`
    (auto-created on first run). The `ssdatabench/` submodule always lives
    under the repo, so it isn't affected by SSDA_ROOT.
    """
    root = os.environ.get("SSDA_ROOT")
    return Path(root) if root else REPO_ROOT


def data_root() -> Path:
    """Where the cleaned survey CSVs live. Override with `SSDA_DATA_ROOT`
    if you want data on a different disk than results."""
    override = os.environ.get("SSDA_DATA_ROOT")
    return Path(override) if override else project_root() / "real_data"


def results_root() -> Path:
    """Where per-experiment outputs go. Override with `SSDA_RESULTS_ROOT`
    if you want results on a different disk than data."""
    override = os.environ.get("SSDA_RESULTS_ROOT")
    return Path(override) if override else project_root() / "results"


def ssdatabench_root() -> Path:
    """The third-party SSDataBench scoring suite, vendored as a git
    submodule at `<repo>/ssdatabench`. Always under the repo — it's code,
    not data — so `git submodule update --init --recursive` is the way to
    get it on a cloud box, not rsync."""
    return REPO_ROOT / "ssdatabench"


@dataclass(frozen=True)
class LLMConfig:
    provider: Literal["openai", "anthropic"]
    base_url: str
    api_key: str
    model: str
    temperature: float = 1.0
    max_tokens: int = 4096


_SENTINEL = object()


def load_llm_config(
    yaml_path: Path | None = None,
    env_path: Path | None | object = _SENTINEL,
    overrides: dict[str, str] | None = None,
) -> LLMConfig:
    """Load LLM configuration with precedence overrides > env > yaml > defaults.

    The API key *value* is only ever read from the environment; only the NAME of
    the variable to read is configurable. That keeps secrets out of every config
    file while letting one batch cycle through providers that authenticate with
    different keys (e.g. OpenRouter's key alongside a first-party one).

    Args:
        yaml_path: yaml file to read, defaults to ``config/llm.yaml``.
        env_path: .env file to load before reading env vars. Pass ``None`` to
            skip loading any .env (useful in tests). Default is the repo root
            ``.env``.
        overrides: per-experiment dict with optional keys
            {provider, base_url, model, api_key_env}. Lets one process run
            experiments back to back against different models without rewriting
            the env or forking processes — useful for the batch runner.
            ``api_key_env`` names the env var holding that provider's key and
            defaults to ``LLM_API_KEY``.
    """
    if env_path is _SENTINEL:
        env_path = REPO_ROOT / ".env"
    if env_path is not None:
        load_dotenv(env_path, override=False)
    yaml_path = yaml_path or DEFAULT_YAML
    data: dict = {}
    if yaml_path.exists():
        with yaml_path.open() as f:
            data = yaml.safe_load(f) or {}

    ov = overrides or {}
    provider = ov.get("provider") or os.environ.get("LLM_PROVIDER", data.get("provider", "openai"))
    base_url = ov.get("base_url") or os.environ.get("LLM_BASE_URL", data.get("base_url", ""))
    model = ov.get("model") or os.environ.get("LLM_MODEL", data.get("model", ""))
    key_env = ov.get("api_key_env") or data.get("api_key_env") or "LLM_API_KEY"
    api_key = os.environ.get(key_env, "")
    temperature = float(data.get("temperature", 1.0))
    max_tokens = int(data.get("max_tokens", 4096))

    if not api_key:
        # Never silently fall back to LLM_API_KEY: sending a first-party key to a
        # third-party base_url would leak it to that provider.
        raise RuntimeError(
            f"{key_env} not found in environment. Set it in .env or export it."
        )
    if provider not in ("openai", "anthropic"):
        raise RuntimeError(f"Unknown LLM_PROVIDER: {provider!r}")

    return LLMConfig(
        provider=provider,  # type: ignore[arg-type]
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
