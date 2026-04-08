#!/usr/bin/env python3
"""
Root inference entrypoint (hackathon / HF submission).

Required environment variables (see also ``.env.example``):
  API_BASE_URL   OpenAI-compatible API base URL (e.g. Groq or OpenAI).
  MODEL_NAME     Chat model id for the OpenAI client.
  HF_TOKEN       API key passed to the OpenAI client (``OPENAI_API_KEY`` is set from this).

Word reports: by default writes under ``./reports`` (override with ``--report-dir`` or
``AUTODATALAB_REPORT_DIR``; disable with ``--no-report``). Requires ``python-docx``
(``pip install -e "./data_cleaning_env[openai]"`` or ``[report]``).

All LLM calls use the ``openai`` Python package (``OpenAI`` client) against ``API_BASE_URL``.

Designed to finish within typical contest limits (e.g. <20 minutes) on modest hardware (e.g. 2 vCPU / 8GB RAM).

Course context: `Building RL Environments with OpenEnv <https://github.com/raun/openenv-course>`_.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _apply_submission_env() -> None:
    """Map submission variable names onto what ``baseline_inference`` / OpenAI client expect."""
    token = (os.environ.get("HF_TOKEN") or "").strip().strip('"').strip("'")
    if token and not (os.environ.get("OPENAI_API_KEY") or "").strip():
        os.environ["OPENAI_API_KEY"] = token
    base = (os.environ.get("API_BASE_URL") or "").strip().rstrip("/")
    if base:
        os.environ["OPENAI_BASE_URL"] = base
        os.environ.setdefault("API_BASE_URL", base)
    model = (os.environ.get("MODEL_NAME") or "").strip()
    if model:
        os.environ["MODEL_NAME"] = model


def main() -> None:
    _apply_submission_env()
    repo = Path(__file__).resolve().parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from data_cleaning_env.baseline_inference import main as baseline_main

    baseline_main()


if __name__ == "__main__":
    main()
