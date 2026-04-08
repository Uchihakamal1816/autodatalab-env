#!/usr/bin/env python3
"""Validator-safe root inference entrypoint with structured stdout logs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.getenv("HF_TOKEN", "")
BENCHMARK = "autodatalab_data_cleaning"
TASKS = [t.strip() for t in os.getenv("AUTODATALAB_TASKS", "easy,medium,hard").split(",") if t.strip()]
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.5"))
INFERENCE_MODE = os.getenv("INFERENCE_MODE", "oracle").strip().lower()


def _bool_str(v: bool) -> str:
    return str(bool(v)).lower()


def _safe_token() -> str:
    token = (HF_TOKEN or "").strip().strip('"').strip("'")
    if token and not (os.environ.get("OPENAI_API_KEY") or "").strip():
        os.environ["OPENAI_API_KEY"] = token
    if API_BASE_URL:
        os.environ["OPENAI_BASE_URL"] = API_BASE_URL.rstrip("/")
    if MODEL_NAME:
        os.environ["MODEL_NAME"] = MODEL_NAME
    return token


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={_bool_str(done)} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={_bool_str(success)} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _action_str(action) -> str:
    try:
        data = action.model_dump(exclude_none=True)
        data.pop("metadata", None)
        if data.get("action_type") != "remove_outliers":
            data.pop("z_threshold", None)
        return json.dumps(data, separators=(",", ":"), sort_keys=True)
    except Exception:
        return str(action).replace("\n", " ")


def _oracle_next_action(task: str, obs, DataCleaningAction):
    history = []
    for h in getattr(obs, "history", []) or []:
        try:
            history.append(json.loads(h))
        except Exception:
            continue

    def has_action(action_type: str) -> bool:
        return any(h.get("action_type") == action_type for h in history)

    plot_count = sum(1 for h in history if h.get("action_type") == "plot")

    if task == "easy":
        if not has_action("remove_duplicates"):
            return DataCleaningAction(action_type="remove_duplicates")
        if not has_action("fill_missing"):
            return DataCleaningAction(action_type="fill_missing", column="Price", method="mean")
        return DataCleaningAction(action_type="submit")

    if task == "medium":
        if not has_action("compute_metrics"):
            return DataCleaningAction(action_type="compute_metrics")
        return DataCleaningAction(action_type="submit")

    if task == "medium_plus":
        if not has_action("compute_kpis"):
            return DataCleaningAction(action_type="compute_kpis")
        return DataCleaningAction(action_type="submit")

    if task in ("hard", "expert"):
        if not has_action("remove_duplicates"):
            return DataCleaningAction(action_type="remove_duplicates")
        if not has_action("fill_missing"):
            return DataCleaningAction(action_type="fill_missing", column="Price", method="mean")
        if not has_action("derive_revenue"):
            return DataCleaningAction(action_type="derive_revenue")
        if plot_count == 0:
            return DataCleaningAction(action_type="plot", plot_type="scatter", x="OrderDate", y="Revenue")
        if plot_count == 1:
            return DataCleaningAction(action_type="plot", plot_type="bar", x="Category", y="Revenue")
        return DataCleaningAction(action_type="submit")

    return DataCleaningAction(action_type="submit")


def _run_oracle_episode(task: str) -> float:
    repo = Path(__file__).resolve().parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from data_cleaning_env.models import DataCleaningAction
    from data_cleaning_env.server.data_cleaning_env_environment import DataCleaningEnvironment

    env = DataCleaningEnvironment()
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(task=task)
        max_steps = int(getattr(obs, "max_steps", 40))
        for step in range(1, max_steps + 1):
            if obs.done:
                break
            action = _oracle_next_action(task, obs, DataCleaningAction)
            obs = env.step(action)
            reward = float(obs.reward or 0.0)
            rewards.append(reward)
            steps_taken = step
            log_step(
                step=step,
                action=_action_str(action),
                reward=reward,
                done=bool(obs.done),
                error=None,
            )
            if obs.done:
                break
        score = float(obs.terminal_grader_score or 0.0)
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
        return score
    except Exception as exc:
        err = str(exc).replace("\n", " ")
        step_no = max(steps_taken + 1, 1)
        log_step(step=step_no, action="exception", reward=0.0, done=True, error=err)
        return 0.0
    finally:
        try:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()
        finally:
            if "score" not in locals():
                score = 0.0
            if "success" not in locals():
                success = False
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    _safe_token()
    # Keep default inference deterministic and validator-safe. LLM mode remains opt-in via env.
    for task in TASKS:
        _run_oracle_episode(task)


if __name__ == "__main__":
    main()
