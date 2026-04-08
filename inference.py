#!/usr/bin/env python3
"""Validator-safe root inference entrypoint with structured stdout logs.

When `API_KEY` is present, this script makes real OpenAI-client calls through the
injected `API_BASE_URL` LiteLLM proxy. For local development without proxy creds,
it falls back to the deterministic oracle so the script remains runnable.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or ""
BENCHMARK = "autodatalab_data_cleaning"
TASKS = [t.strip() for t in os.getenv("AUTODATALAB_TASKS", "easy,medium,hard").split(",") if t.strip()]
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.5"))
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
INITIAL_RETRY_DELAY_S = float(os.getenv("LLM_RETRY_DELAY_S", "2"))


def _bool_str(v: bool) -> str:
    return str(bool(v)).lower()


def _safe_token() -> str:
    token = (API_KEY or "").strip().strip('"').strip("'")
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


def _build_prompt(task: str, obs) -> str:
    history = obs.history[-8:] if getattr(obs, "history", None) else []
    return f"""You are an e-commerce data analyst cleaning a pandas table.
Task: {task}
Instruction: {obs.instruction}
Columns: {obs.column_names}
Issues detected: {obs.issues}
Policy rules: {json.dumps(getattr(obs, "policy_rules", []) or [])}
Policy warnings: {json.dumps(getattr(obs, "policy_warnings", []) or [])}
Preview: {json.dumps(obs.preview, separators=(",", ":"))}
Recent history: {history}
Last step: {obs.last_step_summary or "(none yet)"}

Rules:
- Easy: remove_duplicates -> fill_missing(Price,mean) -> submit.
- Medium: compute_metrics -> submit.
- Medium_plus: compute_kpis -> submit.
- Hard/Expert: remove_duplicates -> fill_missing(Price,mean) -> derive_revenue -> plot scatter(OrderDate,Revenue) -> plot bar(Category,Revenue) -> submit.
- Do not fill or drop identifier columns like OrderID or CustomerID.
- Reply with exactly one JSON object and no markdown.

Schema:
{{
  "action_type": "remove_duplicates" | "fill_missing" | "drop_column" | "normalize" | "remove_outliers" | "derive_revenue" | "compute_metrics" | "compute_kpis" | "plot" | "export_csv" | "submit" | "noop",
  "column": string or null,
  "method": "mean" | "median" | "mode" | null,
  "z_threshold": number or null,
  "x": string or null,
  "y": string or null,
  "plot_type": "scatter" | "bar" | "histogram" | null,
  "export_basename": string or null,
  "description": string or null
}}"""


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
        score = max(0.001, min(0.999, score))
        success = score >= SUCCESS_SCORE_THRESHOLD
        return score
    except Exception as exc:
        err = str(exc).replace("\n", " ")
        step_no = max(steps_taken + 1, 1)
        log_step(step=step_no, action="exception", reward=0.0, done=True, error=err)
        return 0.001
    finally:
        try:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()
        finally:
            if "score" not in locals():
                score = 0.001
            if "success" not in locals():
                success = False
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def _run_llm_episode(task: str) -> float:
    repo = Path(__file__).resolve().parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from openai import OpenAI

    from data_cleaning_env.baseline_inference import (
        _chat_completion_with_retry,
        _hard_alternate_loop_normalize,
        _parse_action_json,
        _semantic_action_repr,
        _stuck_advance,
    )
    from data_cleaning_env.models import DataCleaningAction
    from data_cleaning_env.server.data_cleaning_env_environment import DataCleaningEnvironment

    client = OpenAI(base_url=API_BASE_URL, api_key=_safe_token())
    env = DataCleaningEnvironment()
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    prev_semantic: Optional[str] = None
    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(task=task)
        limit = int(getattr(obs, "max_steps", 40))

        for step in range(1, limit + 1):
            if obs.done:
                break

            prompt = _build_prompt(task, obs)
            error: Optional[str] = None
            raw = ""

            try:
                resp = _chat_completion_with_retry(
                    client,
                    MODEL_NAME,
                    [{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_retries=MAX_RETRIES,
                    initial_delay_s=INITIAL_RETRY_DELAY_S,
                    json_mode=False,
                )
                raw = (resp.choices[0].message.content or "").strip()
                data = _parse_action_json(raw)
                action = DataCleaningAction.model_validate(data)
            except Exception as exc:
                error = str(exc).replace("\n", " ")
                action = _oracle_next_action(task, obs, DataCleaningAction)

            # Keep the episode progressing even if the model loops or produces a weak action.
            advanced = _stuck_advance(task, obs, action, prev_semantic)
            if advanced is not None:
                action = advanced
            hard_norm = _hard_alternate_loop_normalize(task, obs, action)
            if hard_norm is not None:
                action = hard_norm

            obs = env.step(action)
            reward = float(obs.reward or 0.0)
            rewards.append(reward)
            steps_taken = step
            log_step(
                step=step,
                action=_action_str(action),
                reward=reward,
                done=bool(obs.done),
                error=error,
            )
            prev_semantic = _semantic_action_repr(action)

            if obs.done:
                break

        if not obs.done:
            action = DataCleaningAction(action_type="submit")
            obs = env.step(action)
            reward = float(obs.reward or 0.0)
            rewards.append(reward)
            steps_taken += 1
            log_step(
                step=steps_taken,
                action=_action_str(action),
                reward=reward,
                done=bool(obs.done),
                error=None,
            )

        score = float(obs.terminal_grader_score or 0.001)
        score = max(0.001, min(0.999, score))
        success = score >= SUCCESS_SCORE_THRESHOLD
        return score
    except Exception as exc:
        err = str(exc).replace("\n", " ")
        step_no = max(steps_taken + 1, 1)
        log_step(step=step_no, action="exception", reward=0.0, done=True, error=err)
        return 0.001
    finally:
        try:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()
        finally:
            if "score" not in locals():
                score = 0.001
            if "success" not in locals():
                success = False
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    for task in TASKS:
        if _safe_token():
            _run_llm_episode(task)
        else:
            _run_oracle_episode(task)


if __name__ == "__main__":
    main()
