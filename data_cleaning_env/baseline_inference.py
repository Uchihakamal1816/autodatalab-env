#!/usr/bin/env python3
"""
Baseline agent for AutoDataLab (tabular data cleaning OpenEnv).

- ``--oracle``: deterministic expert policy (reproducible 1.0 on bundled tasks).
- LLM mode uses the OpenAI Python client with compatible endpoints for:
  **OpenAI**, **Groq**, or **Google Gemini** (see env vars below).

Prints per-task terminal grader scores in [0, 1] and their mean.

When ``AUTODATALAB_CSV_DIR`` or ``AUTODATALAB_PLOT_DIR`` is set, writes ``{task}_final.csv``
after each task (final working table).

Word (.docx) reports go to ``./reports`` by default (override with ``--report-dir`` or
``AUTODATALAB_REPORT_DIR``; skip with ``--no-report``). Requires ``python-docx``: session summary
plus per-task operations, table preview, and issue flags.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

from data_cleaning_env.episode_report import (
    EpisodeTrace,
    EpisodeTraceBuilder,
    write_episode_docx,
    write_session_docx,
)
from data_cleaning_env.models import DataCleaningAction, DataCleaningObservation
from data_cleaning_env.server.data_cleaning_env_environment import DataCleaningEnvironment

# Gemini OpenAI-compatible API (Google AI Studio / Generative Language API)
_DEFAULT_GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _baseline_report_dir() -> str:
    """Directory for Word reports: ``--report-dir`` or ``AUTODATALAB_REPORT_DIR``."""
    d = (os.environ.get("AUTODATALAB_REPORT_DIR") or "").strip()
    return d


def _baseline_export_csv_root() -> str:
    """Directory for per-task CSV snapshots: ``AUTODATALAB_CSV_DIR``, else ``AUTODATALAB_PLOT_DIR``."""
    d = (os.environ.get("AUTODATALAB_CSV_DIR") or "").strip()
    if d:
        return d
    return (os.environ.get("AUTODATALAB_PLOT_DIR") or "").strip()


def _save_baseline_task_csv(
    env: DataCleaningEnvironment,
    task: str,
    *,
    verbose: bool = False,
) -> None:
    """After each task, write ``{task}_final.csv`` when an export directory is configured."""
    root = _baseline_export_csv_root()
    if not root:
        return
    try:
        path = Path(root).expanduser().resolve() / f"{task}_final.csv"
        env.write_working_table_csv(path)
        if verbose:
            print(f"  [{task}] saved table → {path}", file=sys.stderr)
    except OSError:
        if verbose:
            print(f"  [{task}] failed to save CSV under {root!r}", file=sys.stderr)


def _oracle_run(
    env: DataCleaningEnvironment,
    task: str,
    *,
    initial_obs: DataCleaningObservation,
    trace: Optional[EpisodeTraceBuilder] = None,
) -> DataCleaningObservation:
    """Expert policy; optional ``trace`` records the same transitions for Word reports."""

    def _step(a: DataCleaningAction) -> DataCleaningObservation:
        nonlocal obs_before
        o = env.step(a)
        if trace is not None:
            trace.add_step(
                obs_before,
                a,
                o,
                parse_ok=True,
                llm_raw=None,
                heuristic_note="oracle expert policy",
            )
        obs_before = o
        return o

    def _clean_pipeline() -> None:
        _step(DataCleaningAction(action_type="remove_duplicates"))
        _step(DataCleaningAction(action_type="fill_missing", column="Price", method="mean"))

    obs_before = initial_obs
    if task == "easy":
        _clean_pipeline()
    elif task == "medium":
        _step(DataCleaningAction(action_type="compute_metrics"))
    elif task == "medium_plus":
        _step(DataCleaningAction(action_type="compute_kpis"))
    elif task == "hard":
        _clean_pipeline()
        _step(DataCleaningAction(action_type="derive_revenue"))
        _step(DataCleaningAction(action_type="compute_revenue_share"))
    elif task == "expert":
        _clean_pipeline()
        _step(DataCleaningAction(action_type="remove_outliers", column="Price", z_threshold=2.5))
        _step(DataCleaningAction(action_type="derive_revenue"))
        _step(DataCleaningAction(action_type="validate_schema"))
        _step(DataCleaningAction(action_type="compute_kpis"))
        _step(DataCleaningAction(action_type="plot", plot_type="scatter", x="OrderDate", y="Revenue"))
        _step(DataCleaningAction(action_type="plot", plot_type="bar", x="Category", y="Revenue"))
        _step(DataCleaningAction(action_type="plot", plot_type="bar", x="Product", y="Revenue"))
    return _step(DataCleaningAction(action_type="submit"))


def _parse_action_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no json object")
    return json.loads(m.group(0))


def _semantic_action_repr(action: DataCleaningAction) -> str:
    d = action.model_dump(exclude_none=True, exclude={"metadata", "description"})
    return json.dumps(d, sort_keys=True)


def _summary_no_progress(summary: str) -> bool:
    """True when the environment reported the last step did not change the table meaningfully."""
    if not summary or "No actions yet" in summary:
        return False
    s = summary.lower()
    if "removed 0 row(s)" in summary:
        return True
    if "remove_duplicates" in s and "removed 0 rows" in summary:
        return True
    if "already had unique" in s:
        return True
    if "noop:" in s and "no operation" in s:
        return True
    if "(filled 0)" in summary or "filled 0." in summary:
        return True
    return False


def _history_has_action_type(obs: DataCleaningObservation, *types: str) -> bool:
    for h in obs.history:
        try:
            d = json.loads(h)
            if d.get("action_type") in types:
                return True
        except Exception:
            continue
    return False


def _plot_count(obs: DataCleaningObservation) -> int:
    n = 0
    for h in obs.history:
        try:
            d = json.loads(h)
            if d.get("action_type") == "plot":
                n += 1
        except Exception:
            continue
    return n


def _fill_missing_count(obs: DataCleaningObservation) -> int:
    n = 0
    for h in obs.history:
        try:
            d = json.loads(h)
            if d.get("action_type") == "fill_missing":
                n += 1
        except Exception:
            continue
    return n


def _hard_pipeline_next_action(obs: DataCleaningObservation) -> Optional[DataCleaningAction]:
    """Next step in the hard-task revenue-share pipeline (for stuck recovery)."""
    if _fill_missing_count(obs) < 1:
        return DataCleaningAction(
            action_type="fill_missing",
            column="Price",
            method="mean",
            description="Auto: impute Price.",
        )
    if not _history_has_action_type(obs, "derive_revenue"):
        return DataCleaningAction(
            action_type="derive_revenue",
            description="Auto: derive Revenue.",
        )
    if not _history_has_action_type(obs, "compute_revenue_share"):
        return DataCleaningAction(
            action_type="compute_revenue_share",
            description="Auto: compute category revenue share %.",
        )
    return DataCleaningAction(
        action_type="submit",
        description="Auto: submit hard task.",
    )


def _expert_pipeline_next_action(obs: DataCleaningObservation) -> Optional[DataCleaningAction]:
    """Next step in the expert-task ecommerce pipeline (for stuck recovery)."""
    if _fill_missing_count(obs) < 1:
        return DataCleaningAction(
            action_type="fill_missing",
            column="Price",
            method="mean",
            description="Auto: impute Price.",
        )
    if not _history_has_action_type(obs, "remove_outliers"):
        return DataCleaningAction(
            action_type="remove_outliers",
            column="Price",
            z_threshold=2.5,
            description="Auto: remove Price outliers.",
        )
    if not _history_has_action_type(obs, "derive_revenue"):
        return DataCleaningAction(
            action_type="derive_revenue",
            description="Auto: derive Revenue.",
        )
    if not _history_has_action_type(obs, "validate_schema"):
        return DataCleaningAction(
            action_type="validate_schema",
            description="Auto: enforce business rules.",
        )
    if not _history_has_action_type(obs, "compute_kpis"):
        return DataCleaningAction(
            action_type="compute_kpis",
            description="Auto: compute KPIs.",
        )
    pc = _plot_count(obs)
    if pc == 0:
        return DataCleaningAction(
            action_type="plot",
            plot_type="scatter",
            x="OrderDate",
            y="Revenue",
            description="Auto: Revenue vs time.",
        )
    if pc == 1:
        return DataCleaningAction(
            action_type="plot",
            plot_type="bar",
            x="Category",
            y="Revenue",
            description="Auto: category revenue bar chart.",
        )
    if pc == 2:
        return DataCleaningAction(
            action_type="plot",
            plot_type="bar",
            x="Product",
            y="Revenue",
            description="Auto: top products bar chart.",
        )
    return DataCleaningAction(
        action_type="submit",
        description="Auto: submit expert task.",
    )


def _stuck_advance(
    task: str,
    obs: DataCleaningObservation,
    action: DataCleaningAction,
    prev_semantic: Optional[str],
) -> Optional[DataCleaningAction]:
    """When the model repeats the same action and the env reported no progress, advance the pipeline."""
    if prev_semantic is None or not _summary_no_progress(obs.last_step_summary):
        return None
    if _semantic_action_repr(action) != prev_semantic:
        return None
    at = action.action_type
    _next_hard = _hard_pipeline_next_action
    _next_expert = _expert_pipeline_next_action
    if at == "remove_outliers":
        if task == "hard":
            return _next_hard(obs)
        if task == "expert":
            return _next_expert(obs)
        return DataCleaningAction(
            action_type="submit",
            description="Auto: submitting (no progress from outlier removal).",
        )
    if at == "remove_duplicates":
        if task == "hard":
            return _next_hard(obs)
        if task == "expert":
            return _next_expert(obs)
        if task == "medium":
            if not _history_has_action_type(obs, "compute_metrics"):
                return DataCleaningAction(
                    action_type="compute_metrics",
                    description="Auto: category revenue metrics.",
                )
            return DataCleaningAction(
                action_type="submit",
                description="Auto: submitting (dedupe loop with no progress).",
            )
        if task == "medium_plus":
            if not _history_has_action_type(obs, "compute_kpis"):
                return DataCleaningAction(
                    action_type="compute_kpis",
                    description="Auto: business KPIs (total revenue, avg order value).",
                )
            return DataCleaningAction(action_type="submit", description="Auto: submitting after KPIs.")
    if at == "fill_missing":
        if task == "hard":
            return _next_hard(obs)
        if task == "expert":
            return _next_expert(obs)
    if at == "compute_metrics" and task == "medium":
        return DataCleaningAction(
            action_type="submit",
            description="Auto: submitting after metrics.",
        )
    if at == "compute_kpis" and task == "medium_plus":
        return DataCleaningAction(action_type="submit", description="Auto: submitting after KPIs.")
    return None


def _hard_alternate_loop_normalize(
    task: str,
    obs: DataCleaningObservation,
    action: DataCleaningAction,
) -> Optional[DataCleaningAction]:
    """
    Hard/expert task: after cleaning + derive_revenue, skip redundant dedupe/outlier loops and
    advance to the next required step or submit.
    """
    if task not in ("hard", "expert"):
        return None
    at = action.action_type
    if at == "submit":
        return None

    if task == "hard":
        if (
            _history_has_action_type(obs, "compute_revenue_share")
            and _history_has_action_type(obs, "derive_revenue")
        ):
            return DataCleaningAction(
                action_type="submit",
                description="Auto: revenue share computed; submitting hard task.",
            )
        if _fill_missing_count(obs) < 1:
            return None
        if not _summary_no_progress(obs.last_step_summary):
            return None
        if at in ("noop",):
            return None
        if at in ("remove_duplicates", "fill_missing", "derive_revenue"):
            return _hard_pipeline_next_action(obs)
        return None

    # expert
    required_plots = 3
    if (
        _plot_count(obs) >= required_plots
        and _history_has_action_type(obs, "derive_revenue")
        and _history_has_action_type(obs, "validate_schema")
    ):
        return DataCleaningAction(
            action_type="submit",
            description="Auto: all plots declared; submitting expert task.",
        )
    if _fill_missing_count(obs) < 1:
        return None
    if not _summary_no_progress(obs.last_step_summary):
        return None
    if at in ("noop",):
        return None
    if at in ("remove_duplicates", "remove_outliers", "fill_missing", "validate_schema"):
        return _expert_pipeline_next_action(obs)
    return None


def _chat_completion_with_retry(
    client: Any,
    model: str,
    messages: list,
    *,
    temperature: float,
    max_retries: int,
    initial_delay_s: float,
    json_mode: bool = False,
) -> Any:
    """Retry on 429 (rate limit / quota burst); Gemini often says 'retry in Ns'."""
    from openai import BadRequestError
    from openai import RateLimitError

    delay = initial_delay_s
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                return client.chat.completions.create(**kwargs)
            except BadRequestError:
                if json_mode and "response_format" in kwargs:
                    kwargs.pop("response_format", None)
                    return client.chat.completions.create(**kwargs)
                raise
        except RateLimitError as e:
            last_err = e
            if attempt >= max_retries - 1:
                raise
            print(
                f"LLM rate limit (429), sleeping {delay:.0f}s then retry "
                f"({attempt + 1}/{max_retries})…",
                file=sys.stderr,
            )
            time.sleep(delay)
            delay = min(delay * 1.5, 120.0)
    raise last_err  # pragma: no cover


def run_llm_episode(
    env: DataCleaningEnvironment,
    task: str,
    client: Any,
    model: str,
    max_steps: int,
    llm_retries: int = 5,
    llm_retry_delay_s: float = 18.0,
    *,
    json_mode: bool = False,
    verbose: bool = False,
    trace_builder: Optional[EpisodeTraceBuilder] = None,
) -> Optional[float]:
    # Rolling window for history shown to LLM: keep last N entries to limit prompt size
    _HISTORY_WINDOW = int(os.getenv("LLM_HISTORY_WINDOW", "8"))

    obs = env.reset(task=task)
    if trace_builder is not None:
        trace_builder.set_initial(obs)
    limit = min(max_steps, obs.max_steps)
    step_i = 0
    prev_semantic: Optional[str] = None

    for _ in range(limit):
        if obs.done:
            _save_baseline_task_csv(env, task, verbose=verbose)
            return obs.terminal_grader_score
        obs_before = obs
        step_i += 1
        # Only send the last N history entries to keep prompt compact and focused
        recent_history = obs.history[-_HISTORY_WINDOW:] if obs.history else []
        history_note = (
            f"(showing last {_HISTORY_WINDOW} of {len(obs.history)} steps)"
            if len(obs.history) > _HISTORY_WINDOW else ""
        )
        prompt = f"""You are an e-commerce data analyst cleaning a pandas table.
Task: {task}
Instruction: {obs.instruction}
Columns: {obs.column_names}
Issues detected: {obs.issues}
Policy rules (always): {json.dumps(getattr(obs, "policy_rules", []) or [])}
Policy warnings (this table): {json.dumps(getattr(obs, "policy_warnings", []) or [])}
Preview (first rows): {json.dumps(obs.preview, indent=0)}
Recent history {history_note}: {recent_history}
Last step (authoritative — what the environment actually did): {obs.last_step_summary or "(none yet)"}

Rules:
- Do not call fill_missing or drop_column on OrderID, CustomerID, or other identifier columns; the environment blocks them.
- Do NOT repeat the same action_type twice in a row unless Last step shows real progress and Issues still require it.
- Easy: dedupe → impute Price → submit.
- Medium: compute_metrics (category revenue) then submit.
- Medium_plus: compute_kpis (TotalRevenue + AvgOrderValue) then submit.
- Hard: dedupe → impute Price → derive_revenue → plot scatter (OrderDate vs Revenue) → plot bar (Category vs Revenue) → submit.
- Expert: same as Hard (full cleaning + derive_revenue + both plots) then submit.
- If Last step says remove_outliers removed 0 row(s), advance the pipeline (do not loop).
- When the instruction is satisfied, submit — do not loop until max_steps.

Reply with ONE JSON object only — no markdown, no code fences — matching this schema:
{{
  "action_type": "remove_duplicates" | "fill_missing" | "drop_column" | "normalize" | "remove_outliers" | "derive_revenue" | "compute_metrics" | "compute_kpis" | "plot" | "export_csv" | "submit" | "noop",
  "column": string or null,
  "method": "mean" | "median" | "mode" | null,
  "z_threshold": number,
  "x": string or null,
  "y": string or null,
  "plot_type": "scatter" | "bar" | "histogram" | null,
  "export_basename": string or null,
  "description": string or null
}}
Use null for unused fields. export_csv writes the current working table to disk when env AUTODATALAB_CSV_DIR is set (optional artifact). Put a short "description" of what you intend; the server reports row index labels after dedupe."""

        resp = _chat_completion_with_retry(
            client,
            model,
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_retries=llm_retries,
            initial_delay_s=llm_retry_delay_s,
            json_mode=json_mode,
        )
        raw = resp.choices[0].message.content or ""
        parse_ok = True
        heuristic_note: Optional[str] = None
        try:
            data = _parse_action_json(raw)
            action = DataCleaningAction.model_validate(data)
        except Exception as e:
            parse_ok = False
            action = DataCleaningAction(action_type="noop")
            if verbose:
                print(f"  [{task} step {step_i}] parse error: {e}", file=sys.stderr)

        if parse_ok:
            hard_norm = _hard_alternate_loop_normalize(task, obs, action)
            if hard_norm is not None:
                action = hard_norm
                h = f"heuristic: hard_alternate_loop_normalize → {action.action_type}"
                heuristic_note = f"{heuristic_note}; {h}" if heuristic_note else h
                if verbose:
                    print(
                        f"  [{task} step {step_i}] heuristic: hard pipeline → {action.action_type}",
                        file=sys.stderr,
                    )

        if parse_ok:
            advanced = _stuck_advance(task, obs, action, prev_semantic)
            if advanced is not None:
                action = advanced
                h = f"heuristic: stuck_advance → {action.action_type}"
                heuristic_note = f"{heuristic_note}; {h}" if heuristic_note else h
                if verbose:
                    print(
                        f"  [{task} step {step_i}] heuristic: same action + no env progress → {action.action_type}",
                        file=sys.stderr,
                    )

        if verbose:
            print(
                f"  [{task} step {step_i}] parse_ok={parse_ok} action={action.model_dump(exclude_none=True)}",
                file=sys.stderr,
            )
            if not parse_ok:
                print(f"  raw (truncated): {raw[:400]!r}", file=sys.stderr)

        obs = env.step(action)
        if trace_builder is not None:
            trace_builder.add_step(
                obs_before,
                action,
                obs,
                parse_ok=parse_ok,
                llm_raw=raw[:6000],
                heuristic_note=heuristic_note,
            )
        prev_semantic = _semantic_action_repr(action)
        if verbose:
            print(f"  [{task} step {step_i}] env: {obs.last_step_summary}", file=sys.stderr)
        if obs.done:
            _save_baseline_task_csv(env, task, verbose=verbose)
            return obs.terminal_grader_score

    obs_before = obs
    obs = env.step(DataCleaningAction(action_type="submit"))
    if trace_builder is not None:
        trace_builder.add_step(
            obs_before,
            DataCleaningAction(action_type="submit"),
            obs,
            parse_ok=True,
            llm_raw=None,
            heuristic_note="forced submit (max_steps reached without submit)",
        )
    _save_baseline_task_csv(env, task, verbose=verbose)
    return obs.terminal_grader_score


def _print_task_score(task: str, terminal_grader: Optional[float]) -> None:
    if terminal_grader is None:
        print(f"task={task} terminal_grader=ungraded (no ground_truth.csv)")
    else:
        print(f"task={task} terminal_grader={terminal_grader:.4f}")


def _emit_word_reports(traces: List[EpisodeTrace], report_root: Path) -> None:
    """Write ``session_report.docx`` plus per-task ``{task}_episode_report.docx``."""
    report_root = Path(report_root).expanduser().resolve()
    try:
        write_session_docx(traces, report_root / "session_report.docx")
        for t in traces:
            write_episode_docx(t, report_root / f"{t.task}_episode_report.docx")
    except ImportError:
        print(
            "Word export skipped: install python-docx (pip install python-docx or pip install -e \".[report]\").",
            file=sys.stderr,
        )


def _resolve_report_dir(args: argparse.Namespace) -> str:
    """Word reports: ``--no-report`` disables; ``--report-dir`` overrides; else env; else ``reports``."""
    if getattr(args, "no_report", False):
        return ""
    if getattr(args, "report_dir", None) is not None:
        s = str(args.report_dir).strip()
        return s  # explicit ``--report-dir ""`` disables
    env = _baseline_report_dir()
    return env if env else "reports"


def _print_mean_graded(scores: List[Optional[float]]) -> None:
    graded = [s for s in scores if s is not None]
    if not graded:
        print("mean_terminal_grader=n/a (no graded tasks)")
    else:
        m = sum(graded) / len(graded)
        print(f"mean_terminal_grader={m:.4f} (over {len(graded)} graded task(s))")


def _try_load_dotenv() -> None:
    """Load `.env` from cwd or parents if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    here = Path(__file__).resolve().parent
    for d in (here, here.parent, here.parent.parent):
        p = d / ".env"
        if p.is_file():
            load_dotenv(p)
            return


def _resolve_llm_settings(args: argparse.Namespace) -> tuple[str, str | None, str]:
    """Return (api_key, base_url, model_id) for OpenAI-compatible providers."""
    openai_key = os.getenv("OPENAI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    hf_token = (os.getenv("HF_TOKEN") or "").strip().strip('"').strip("'")
    if openai_key:
        openai_key = openai_key.strip().strip('"').strip("'")
    if groq_key:
        groq_key = groq_key.strip().strip('"').strip("'")
    if gemini_key:
        gemini_key = gemini_key.strip().strip('"').strip("'")

    raw_provider = (getattr(args, "provider", None) or os.getenv("LLM_PROVIDER") or "auto").lower().strip()
    provider = raw_provider if raw_provider in ("auto", "groq", "gemini", "openai") else "auto"

    if provider == "gemini":
        api_key = gemini_key
    elif provider == "groq":
        api_key = groq_key or (
            openai_key if (openai_key and openai_key.startswith("gsk_")) else None
        )
    elif provider == "openai":
        api_key = openai_key
    elif provider == "auto":
        # Stale Groq key often left in OPENAI_API_KEY; prefer Gemini when both are set
        if gemini_key and openai_key and openai_key.startswith("gsk_"):
            api_key = gemini_key
        else:
            api_key = openai_key or groq_key or gemini_key
    else:
        api_key = openai_key or groq_key or gemini_key

    using_gemini = bool(gemini_key and api_key == gemini_key)

    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("API_BASE_URL")
    if base_url:
        base_url = base_url.strip().rstrip("/")

    if provider == "gemini" or using_gemini:
        gem_override = os.getenv("GEMINI_OPENAI_BASE_URL")
        if gem_override:
            base_url = gem_override.strip()
        elif base_url and "groq.com" in base_url:
            base_url = None
        if not base_url:
            base_url = _DEFAULT_GEMINI_OPENAI_BASE
    elif not base_url and api_key and api_key.startswith("gsk_"):
        base_url = "https://api.groq.com/openai/v1"
    elif not base_url and groq_key and not openai_key:
        base_url = "https://api.groq.com/openai/v1"

    model = args.model or os.getenv("MODEL_NAME")
    if not model:
        if using_gemini or provider == "gemini":
            # 1.5-flash often has free-tier quota when 2.0-flash shows limit:0
            model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        elif groq_key and not openai_key:
            model = "llama-3.1-8b-instant"
        elif api_key and api_key.startswith("gsk_"):
            model = "llama-3.1-8b-instant"
        else:
            model = "gpt-4o-mini"
    elif using_gemini and not args.model and model and not str(model).startswith("gemini"):
        # MODEL_NAME from .env was for Groq (e.g. llama-*) but we switched to Gemini in auto
        model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    if not api_key and hf_token:
        api_key = hf_token

    return api_key, base_url, model


def _parse_tasks_list(raw: str) -> List[str]:
    """Validate ``--tasks`` (comma-separated easy/medium/medium_plus/hard/expert)."""
    allowed = ("easy", "medium", "medium_plus", "hard", "expert")
    seen: set[str] = set()
    out: List[str] = []
    for part in raw.split(","):
        t = part.strip().lower()
        if not t:
            continue
        if t not in allowed:
            raise SystemExit(f"--tasks: unknown task {t!r}; use one or more of: {', '.join(allowed)}")
        if t not in seen:
            seen.add(t)
            out.append(t)
    if not out:
        raise SystemExit("--tasks: empty; example: --tasks easy or --tasks easy,hard")
    return out


def main() -> None:
    _try_load_dotenv()

    parser = argparse.ArgumentParser(description="AutoDataLab baseline inference")
    parser.add_argument("--oracle", action="store_true", help="Deterministic expert policy")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Chat model id (defaults: gpt-4o-mini | llama-3.1-8b-instant for Groq | "
            "gemini-1.5-flash for Gemini — override with MODEL_NAME / GEMINI_MODEL)"
        ),
    )
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument(
        "--llm-retries",
        type=int,
        default=int(os.getenv("LLM_MAX_RETRIES", "5")),
        help="Retries per step on HTTP 429 (rate limit). Default 5.",
    )
    parser.add_argument(
        "--llm-retry-delay",
        type=float,
        default=float(os.getenv("LLM_RETRY_DELAY_S", "18")),
        help="Initial seconds to wait before retrying after 429. Default 18.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print each LLM step: parsed action and parse failures (to stderr).",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not write Word (.docx) session / per-task reports.",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory for Word (.docx) reports (session + per-task). "
            "Default: ./reports if AUTODATALAB_REPORT_DIR is unset. "
            "Requires python-docx (pip install -e \".[report]\" or \".[openai]\")."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "groq", "gemini", "openai"],
        default=None,
        help=(
            "Which API credential to use: groq=Groq, gemini=GEMINI_API_KEY / GOOGLE_API_KEY, "
            "openai=OPENAI_API_KEY. "
            "auto order: OPENAI → GROQ → GEMINI; if OPENAI is gsk_* and GEMINI_API_KEY is set, Gemini wins."
        ),
    )
    parser.add_argument(
        "--tasks",
        default="easy,medium,hard",
        metavar="LIST",
        help=(
            "Comma-separated episodes to run: easy, medium, medium_plus, hard, expert "
            "(default: easy,medium,hard). Use e.g. --tasks easy for a quick smoke test."
        ),
    )
    args = parser.parse_args()

    tasks = _parse_tasks_list(args.tasks)
    scores: List[Optional[float]] = []

    if args.oracle:
        report_dir = _resolve_report_dir(args)
        traces_pdf: List[EpisodeTrace] = []
        for task in tasks:
            env = DataCleaningEnvironment()
            obs0 = env.reset(task=task)
            tb: Optional[EpisodeTraceBuilder] = None
            if report_dir:
                tb = EpisodeTraceBuilder(task, "oracle", model_name=None)
                tb.set_initial(obs0)
            final = _oracle_run(env, task, initial_obs=obs0, trace=tb)
            _save_baseline_task_csv(env, task, verbose=args.verbose)
            sc = final.terminal_grader_score
            scores.append(sc)
            _print_task_score(task, sc)
            if tb is not None:
                traces_pdf.append(tb.build(sc, env=env))
        if report_dir and traces_pdf:
            _emit_word_reports(traces_pdf, Path(report_dir))
            print(f"Word reports written under {report_dir!r}", file=sys.stderr)
        _print_mean_graded(scores)
        return

    api_key, base_url, model = _resolve_llm_settings(args)
    if not api_key:
        prov = (args.provider or os.getenv("LLM_PROVIDER") or "auto").lower()
        if prov == "gemini":
            print(
                "Set GEMINI_API_KEY or GOOGLE_API_KEY (Google AI Studio / Gemini API).",
                file=sys.stderr,
            )
        else:
            print(
                "Set OPENAI_API_KEY, GROQ_API_KEY, or GEMINI_API_KEY / GOOGLE_API_KEY (or use --oracle)",
                file=sys.stderr,
            )
        sys.exit(1)

    try:
        from openai import OpenAI
        from openai import AuthenticationError as OpenAIAuthError
        from openai import RateLimitError as OpenAIRateLimitError
    except ImportError as e:
        raise SystemExit("pip install openai") from e

    client = OpenAI(api_key=api_key, base_url=base_url)
    print(
        f"LLM: model={model!r} base_url={base_url or '(OpenAI default)'} "
        f"(provider={args.provider or os.getenv('LLM_PROVIDER') or 'auto'})",
        file=sys.stderr,
    )

    report_dir = _resolve_report_dir(args)
    traces_pdf: List[EpisodeTrace] = []
    try:
        for task in tasks:
            env = DataCleaningEnvironment()
            use_json = os.getenv("LLM_JSON_MODE", "1").lower() not in ("0", "false", "no")
            groq_host = base_url and "groq.com" in base_url
            tb: Optional[EpisodeTraceBuilder] = None
            if report_dir:
                tb = EpisodeTraceBuilder(task, "llm", model_name=model)
            sc = run_llm_episode(
                env,
                task,
                client,
                model,
                args.max_steps,
                llm_retries=args.llm_retries,
                llm_retry_delay_s=args.llm_retry_delay,
                json_mode=bool(use_json and groq_host),
                verbose=args.verbose,
                trace_builder=tb,
            )
            scores.append(sc)
            _print_task_score(task, sc)
            if tb is not None:
                traces_pdf.append(tb.build(sc, env=env))
        if report_dir and traces_pdf:
            _emit_word_reports(traces_pdf, Path(report_dir))
            print(f"Word reports written under {report_dir!r}", file=sys.stderr)
    except OpenAIAuthError as err:
        print(
            "\nAuthentication failed (401). Check that your API key is valid.\n"
            "- Groq: https://console.groq.com/keys (gsk_)\n"
            "- Gemini: https://aistudio.google.com/apikey (GEMINI_API_KEY or GOOGLE_API_KEY)\n"
            "- OpenAI: platform.openai.com\n"
            "\nIf you use Gemini but have an old gsk_ in OPENAI_API_KEY, run:\n"
            "  python baseline_inference.py --provider gemini\n"
            "or remove/comment OPENAI_API_KEY in .env so auto picks GEMINI_API_KEY.\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from err
    except OpenAIRateLimitError as err:
        print(
            "\nRate limit / quota (429) from the LLM provider.\n"
            "- Gemini: free tier may have no quota for some models (error often shows limit:0). "
            "Try:  export GEMINI_MODEL=gemini-1.5-flash\n"
            "  Or enable billing / check https://ai.google.dev/gemini-api/docs/rate-limits\n"
            "- Wait a few minutes and retry, or use Groq: --provider groq with GROQ_API_KEY\n"
            "- This script retries 429s automatically; increase delay: --llm-retry-delay 30\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from err

    _print_mean_graded(scores)


if __name__ == "__main__":
    main()
