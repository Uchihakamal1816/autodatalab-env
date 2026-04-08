# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tabular data-cleaning environment (pandas-backed, real-world ETL-style task)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
import pandas as pd
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata, State

try:
    from ..graders import grade_task
    from ..models import (
        DEFAULT_POLICY_RULES,
        DataCleaningAction,
        DataCleaningObservation,
        EpisodeReward,
    )
except ImportError:
    from graders import grade_task
    from models import (
        DEFAULT_POLICY_RULES,
        DataCleaningAction,
        DataCleaningObservation,
        EpisodeReward,
    )

# Compare actions for repeat penalty without LLM-only fields (e.g. description).
_SEMANTIC_ACTION_EXCLUDE = frozenset({"metadata", "description"})

# Columns treated as identifiers — not imputed or dropped without explicit policy (blocked here).
_IDENTIFIER_EXACT = frozenset(
    {
        "name",
        "id",
        "email",
        "ssn",
        "phone",
        "candidate",
        "candidate_id",
        "orderid",
        "customerid",
    }
)


def _is_identifier_column(column: str) -> bool:
    c = column.strip().lower()
    if c in _IDENTIFIER_EXACT:
        return True
    if "name" in c:
        return True
    if c.endswith("_id"):
        return True
    return False


def _find_expiry_column(columns: Any) -> Optional[str]:
    for col in columns:
        cl = str(col).strip().lower()
        if cl in {"expirydays", "expiry_days", "offerexpirydays", "offer_expiry_days"}:
            return str(col)
    return None


def _detect_policy_warnings(df: pd.DataFrame) -> List[str]:
    """Governance-style flags separate from generic `issues` (e.g. heuristic quality tags)."""
    out: List[str] = []
    expiry_col = _find_expiry_column(df.columns)
    if expiry_col is not None:
        s = pd.to_numeric(df[expiry_col], errors="coerce")
        ok = s.dropna()
        if len(ok) and ((ok < 0) | (ok > 365)).any():
            out.append(
                "expiry_implausible: some ExpiryDays values are outside a practical range (0–365); "
                "verify catalog data before trusting downstream commerce analytics."
            )
    return out


def _semantic_action_repr(action: DataCleaningAction) -> str:
    d = action.model_dump(exclude_none=True, exclude=_SEMANTIC_ACTION_EXCLUDE)
    return json.dumps(d, sort_keys=True)


TASK_SPECS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "difficulty": "easy",
        "max_steps": 22,
        "instruction": (
            "E-commerce Task 1 (Data cleaning): remove duplicate orders, then impute missing Price "
            "using mean. ExpiryDays may appear in warnings for data quality review, but it is not part of the "
            "required transformation. Do not impute OrderID or CustomerID. Call submit when the table "
            "matches the analyst reference."
        ),
    },
    "medium": {
        "difficulty": "medium",
        "max_steps": 18,
        "instruction": (
            "E-commerce Task 2 (Business metrics): from order-level data, run compute_metrics to build "
            "Category-level total revenue (Price x Quantity per row, summed by Category). "
            "Then submit."
        ),
    },
    "hard": {
        "difficulty": "hard",
        "max_steps": 38,
        "instruction": (
            "E-commerce Task 3 (Insight + visualization): clean orders (dedupe, impute Price), "
            "derive Revenue = Price x Quantity, then declare a scatter plot OrderDate vs Revenue and a bar chart "
            "Category vs Revenue. Then submit."
        ),
    },
    "medium_plus": {
        "difficulty": "medium",
        "max_steps": 22,
        "instruction": (
            "E-commerce Task 2+ (Full business metrics): from clean order-level data, run compute_kpis to produce "
            "a summary table with rows: Metric (TotalRevenue, AvgOrderValue) and Value. Then submit."
        ),
    },
    "expert": {
        "difficulty": "hard",
        "max_steps": 50,
        "instruction": (
            "E-commerce Task 4 (Expert pipeline): full cleaning (dedupe, impute Price), derive Revenue, "
            "declare scatter OrderDate vs Revenue and bar Category vs Revenue, then submit."
        ),
    },
}


def _tasks_root() -> Path:
    return Path(__file__).resolve().parent.parent / "tasks"


class DataCleaningEnvironment(Environment[DataCleaningAction, DataCleaningObservation, State]):
    """Stateful data-cleaning episode with deterministic grading."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._episode_id: str = str(uuid4())
        self._step_count: int = 0
        self._task_name: str = "easy"
        self._df: pd.DataFrame = pd.DataFrame()
        self._gt: pd.DataFrame = pd.DataFrame()
        self._grading_enabled: bool = True
        self._instruction_text: str = ""
        self._metadata: Dict[str, Any] = {}
        self._history: List[str] = []
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._last_plot: Optional[Dict[str, Any]] = None
        self._plot_history: List[Dict[str, Any]] = []
        self._prev_action_repr: Optional[str] = None
        self._last_step_summary: str = ""

    def _load_task(self, task_name: str) -> None:
        root = _tasks_root() / task_name
        raw_path = root / "raw.csv"
        if not raw_path.is_file():
            raise FileNotFoundError(f"Task {task_name!r} missing raw.csv: {raw_path}")
        self._df = pd.read_csv(raw_path).copy()
        gt_path = root / "ground_truth.csv"
        if gt_path.is_file():
            self._gt = pd.read_csv(gt_path)
            self._grading_enabled = True
        else:
            self._gt = pd.DataFrame()
            self._grading_enabled = False
        meta_path = root / "metadata.json"
        if meta_path.is_file():
            with open(meta_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
        else:
            self._metadata = {}
        self._task_name = task_name
        spec = TASK_SPECS.get(task_name, TASK_SPECS["easy"])
        base = spec["instruction"]
        if not self._grading_enabled:
            self._instruction_text = (
                base
                + " (This task has no ground_truth.csv; terminal score is not computed.)"
            )
        else:
            self._instruction_text = base

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: str = "easy",
        **kwargs: Any,
    ) -> DataCleaningObservation:
        self._reset_rubric()
        self._episode_id = episode_id or str(uuid4())
        self._step_count = 0
        self._history = []
        self._cumulative_reward = 0.0
        self._done = False
        self._last_plot = None
        self._plot_history = []
        self._prev_action_repr = None
        self._last_step_summary = "No actions yet — table is the loaded raw.csv."

        task = (task or "easy").lower().strip()
        if task not in TASK_SPECS:
            task = "easy"
        self._load_task(task)

        if seed is not None:
            # Reserved for future stochastic variants
            pass

        return self._observe(initial=True)

    def _detect_issues(self) -> List[str]:
        issues: List[str] = []
        if self._df.isnull().sum().sum() > 0:
            issues.append("missing_values")
        if self._df.duplicated().sum() > 0:
            issues.append("duplicates")
        num_cols = self._df.select_dtypes(include=["number"]).columns
        for col in num_cols:
            if str(col).strip().lower() in {"expirydays", "expiry_days"}:
                continue
            s = self._df[col].dropna()
            if len(s) < 3:
                continue
            z = (s - s.mean()).abs() / (s.std() if s.std() and s.std() > 0 else 1.0)
            if (z > 3).any():
                issues.append(f"numeric_outliers:{col}")
                break
        return issues

    def _observe(self, initial: bool = False) -> DataCleaningObservation:
        spec = TASK_SPECS[self._task_name]
        issues = self._detect_issues()
        rb = None
        if not initial:
            rb = EpisodeReward(
                immediate=0.0,
                cumulative=self._cumulative_reward,
                terminal_grader=None,
            )
        return DataCleaningObservation(
            done=self._done,
            reward=0.0 if initial else None,
            preview=self._df.head(8).to_dict(orient="records"),
            column_names=list(self._df.columns),
            issues=issues,
            policy_rules=list(DEFAULT_POLICY_RULES),
            policy_warnings=_detect_policy_warnings(self._df),
            task_name=self._task_name,
            task_difficulty=spec["difficulty"],
            max_steps=spec["max_steps"],
            history=list(self._history),
            cumulative_reward=self._cumulative_reward,
            reward_breakdown=rb,
            terminal_grader_score=None,
            instruction=self._instruction_text,
            last_step_summary=self._last_step_summary,
        )

    def _safe_normalize(self, col: str) -> bool:
        c = pd.to_numeric(self._df[col], errors="coerce")
        lo, hi = c.min(), c.max()
        if pd.isna(lo) or (hi - lo) == 0:
            return False
        self._df[col] = (c - lo) / (hi - lo)
        return True

    def _try_save_plot_png(self) -> None:
        """If AUTODATALAB_PLOT_DIR is set and matplotlib is installed, write a PNG for the last plot action."""
        root = os.environ.get("AUTODATALAB_PLOT_DIR", "").strip()
        if not root or self._last_plot is None:
            return
        try:
            try:
                from ..plot_artifacts import save_plot_to_png
            except ImportError:
                from plot_artifacts import save_plot_to_png
        except ImportError:
            return
        try:
            out = Path(root) / f"{self._episode_id}_step{self._step_count}.png"
            save_plot_to_png(
                self._df,
                self._last_plot.get("plot_type"),
                self._last_plot.get("x"),
                self._last_plot.get("y"),
                out,
                title=f"episode {self._episode_id[:8]}… step {self._step_count}",
            )
        except Exception:
            # Optional artifact path must not break episodes
            pass

    def _try_write_export_csv(self, action: DataCleaningAction) -> Optional[Path]:
        """If AUTODATALAB_CSV_DIR is set, write the current table to a CSV file."""
        root = os.environ.get("AUTODATALAB_CSV_DIR", "").strip()
        if not root:
            return None
        try:
            stem_in = (action.export_basename or "").strip()
            if stem_in:
                stem = re.sub(r"[^a-zA-Z0-9_.-]+", "_", stem_in)[:80] or "export"
                name = f"{self._episode_id}_step{self._step_count}_{stem}.csv"
            else:
                name = f"{self._episode_id}_step{self._step_count}.csv"
            out_dir = Path(root).expanduser().resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / name
            self._df.to_csv(path, index=False)
            return path
        except Exception:
            return None

    def _remove_outliers_col(self, col: str, z_thr: float) -> int:
        """Uses log1p + modified z-score (MAD) so single bogus values are detectable on small tables."""
        before = len(self._df)
        s = pd.to_numeric(self._df[col], errors="coerce")
        if s.notna().sum() < 3:
            return 0
        t = np.log1p(s.clip(lower=0))
        med = float(np.nanmedian(t))
        mad = float(np.nanmedian(np.abs(t - med)))
        if mad == 0 or np.isnan(mad):
            return 0
        mz = 0.6745 * np.abs(t - med) / mad
        # Keep rows with missing values in this column (impute later); drop only flagged non-nulls.
        mask = (mz <= z_thr) | s.isna()
        self._df = self._df.loc[mask].reset_index(drop=True)
        return before - len(self._df)

    def step(
        self,
        action: DataCleaningAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> DataCleaningObservation:
        if self._done:
            return self._terminal_observation(repeat=True)

        self._step_count += 1
        spec = TASK_SPECS[self._task_name]
        max_steps = spec["max_steps"]
        immediate = 0.0
        err = False

        ar = action.model_dump(exclude_none=True, exclude={"metadata"})
        self._history.append(json.dumps(ar, sort_keys=True))

        # Loop / noop penalties (semantic action only — ignore LLM description text)
        rep = _semantic_action_repr(action)
        if self._prev_action_repr == rep and action.action_type not in ("noop",):
            immediate -= 0.06
        self._prev_action_repr = rep

        step_summary = ""

        try:
            if action.action_type == "remove_duplicates":
                before = len(self._df)
                removed_labels = self._df.index[self._df.duplicated(keep="first")].tolist()
                self._df = self._df.drop_duplicates().reset_index(drop=True)
                removed = before - len(self._df)
                if removed:
                    immediate += 0.14
                    lab = (
                        str(removed_labels)
                        if len(removed_labels) <= 32
                        else str(removed_labels[:32]) + "…"
                    )
                    step_summary = (
                        f"Removed {removed} duplicate row(s) (row count {before}→{len(self._df)}). "
                        f"Dropped index labels (0-based, original table order): {lab}"
                    )
                else:
                    step_summary = "remove_duplicates: removed 0 rows (table already had unique rows)."

            elif action.action_type == "fill_missing":
                if not action.column or action.method not in ("mean", "median", "mode"):
                    immediate -= 0.05
                    step_summary = "fill_missing: invalid column or method (no change)."
                else:
                    col = action.column
                    if col not in self._df.columns:
                        err = True
                        step_summary = f"fill_missing: column {col!r} not found."
                    elif _is_identifier_column(col):
                        immediate -= 0.08
                        step_summary = (
                            f"fill_missing: policy — column {col!r} is treated as an identifier; "
                            "imputation not applied (define domain rules in production)."
                        )
                    else:
                        before_nulls = int(self._df[col].isnull().sum())
                        s = self._df[col]
                        if action.method == "mean":
                            fill = pd.to_numeric(s, errors="coerce").mean()
                        elif action.method == "median":
                            fill = pd.to_numeric(s, errors="coerce").median()
                        else:
                            fill = s.mode(dropna=True).iloc[0] if s.mode(dropna=True).size else None
                        if isinstance(fill, (int, float, np.integer, np.floating)) and not pd.isna(fill):
                            fill = round(float(fill), 2)
                        self._df[col] = s.fillna(fill)
                        after_nulls = int(self._df[col].isnull().sum())
                        filled = before_nulls - after_nulls
                        if after_nulls < before_nulls:
                            immediate += 0.14
                        step_summary = (
                            f"fill_missing on {col!r} using {action.method}: "
                            f"null cells {before_nulls}→{after_nulls} (filled {filled})."
                        )

            elif action.action_type == "drop_column":
                if not action.column or action.column not in self._df.columns:
                    immediate -= 0.08
                    step_summary = "drop_column: missing or unknown column (no drop)."
                elif _is_identifier_column(action.column):
                    immediate -= 0.08
                    step_summary = (
                        f"drop_column: policy — column {action.column!r} is treated as an identifier; "
                        "drop not applied (define domain rules in production)."
                    )
                else:
                    c = action.column
                    self._df.drop(columns=[c], inplace=True)
                    immediate -= 0.12
                    step_summary = f"drop_column: removed column {c!r}."

            elif action.action_type == "normalize":
                if not action.column or action.column not in self._df.columns:
                    immediate -= 0.05
                    step_summary = "normalize: column missing (no change)."
                elif self._safe_normalize(action.column):
                    immediate += 0.08
                    step_summary = f"normalize: scaled column {action.column!r} to [0, 1]."
                else:
                    immediate -= 0.03
                    step_summary = f"normalize: could not scale {action.column!r} (constant or non-numeric)."

            elif action.action_type == "remove_outliers":
                if not action.column or action.column not in self._df.columns:
                    immediate -= 0.05
                    step_summary = "remove_outliers: column missing (no change)."
                else:
                    before_rows = len(self._df)
                    removed = self._remove_outliers_col(action.column, action.z_threshold)
                    if removed > 0:
                        immediate += 0.14
                    else:
                        immediate -= 0.02
                    step_summary = (
                        f"remove_outliers on {action.column!r} (MAD/z≤{action.z_threshold}): "
                        f"removed {removed} row(s); rows {before_rows}→{len(self._df)}."
                    )

            elif action.action_type == "derive_revenue":
                if "Price" not in self._df.columns or "Quantity" not in self._df.columns:
                    immediate -= 0.06
                    step_summary = "derive_revenue: need Price and Quantity columns."
                else:
                    p = pd.to_numeric(self._df["Price"], errors="coerce")
                    q = pd.to_numeric(self._df["Quantity"], errors="coerce")
                    self._df["Revenue"] = p * q
                    immediate += 0.1
                    step_summary = "derive_revenue: Revenue = Price * Quantity (per row)."

            elif action.action_type == "compute_metrics":
                df = self._df
                if not all(c in df.columns for c in ("Price", "Quantity", "Category")):
                    immediate -= 0.08
                    step_summary = "compute_metrics: need Price, Quantity, and Category columns."
                else:
                    t = df.copy()
                    t["Price"] = pd.to_numeric(t["Price"], errors="coerce")
                    t["Quantity"] = pd.to_numeric(t["Quantity"], errors="coerce")
                    t["_rev"] = t["Price"] * t["Quantity"]
                    agg = t.groupby("Category", as_index=False)["_rev"].sum().rename(columns={"_rev": "Revenue"})
                    self._df = agg.sort_values("Category").reset_index(drop=True)
                    immediate += 0.18
                    step_summary = f"compute_metrics: revenue by Category ({len(self._df)} rows)."

            elif action.action_type == "compute_kpis":
                df = self._df
                if not all(c in df.columns for c in ("Price", "Quantity")):
                    immediate -= 0.08
                    step_summary = "compute_kpis: need Price and Quantity columns."
                else:
                    t = df.copy()
                    t["Price"] = pd.to_numeric(t["Price"], errors="coerce")
                    t["Quantity"] = pd.to_numeric(t["Quantity"], errors="coerce")
                    t["_rev"] = t["Price"] * t["Quantity"]
                    total_rev = float(t["_rev"].sum())
                    # avg order value = total revenue / unique orders
                    if "OrderID" in t.columns:
                        n_orders = t["OrderID"].nunique()
                    else:
                        n_orders = len(t)
                    avg_ov = total_rev / n_orders if n_orders > 0 else 0.0
                    self._df = pd.DataFrame({
                        "Metric": ["TotalRevenue", "AvgOrderValue"],
                        "Value": [round(total_rev, 2), round(avg_ov, 2)],
                    })
                    immediate += 0.18
                    step_summary = (
                        f"compute_kpis: TotalRevenue={total_rev:,.2f}, "
                        f"AvgOrderValue={avg_ov:,.2f} ({n_orders} unique orders)."
                    )

            elif action.action_type == "plot":
                self._last_plot = {
                    "plot_type": action.plot_type,
                    "x": action.x,
                    "y": action.y,
                }
                self._plot_history.append(dict(self._last_plot))
                self._try_save_plot_png()
                immediate += 0.06
                step_summary = (
                    f"plot: recorded {action.plot_type!r} with x={action.x!r}, y={action.y!r}."
                )

            elif action.action_type == "export_csv":
                path = self._try_write_export_csv(action)
                if path is not None:
                    immediate += 0.05
                    step_summary = (
                        f"export_csv: wrote {len(self._df)} rows to {path}."
                    )
                else:
                    step_summary = (
                        "export_csv: set env AUTODATALAB_CSV_DIR to save the working table as CSV."
                    )

            elif action.action_type == "submit":
                self._done = True
                if self._grading_enabled:
                    step_summary = "submit: episode closed for grading."
                else:
                    step_summary = (
                        "submit: episode closed (no ground_truth.csv — no terminal score)."
                    )

            elif action.action_type == "noop":
                immediate -= 0.03
                step_summary = "noop: no operation applied."

        except Exception:
            err = True
            immediate -= 0.15
            step_summary = "error: step raised an exception (table may be unchanged)."

        if err:
            immediate -= 0.05

        # Forced termination
        if self._step_count >= max_steps and not self._done:
            self._done = True

        self._cumulative_reward += immediate

        terminal_grader: Optional[float] = None
        if self._done:
            terminal_grader = grade_task(
                self._df,
                self._gt,
                self._metadata,
                self._last_plot,
                self._plot_history,
            )
            # Terminal reward combines shaping with graded outcome (skip if ungraded)
            if terminal_grader is not None:
                immediate += terminal_grader

        if not step_summary:
            step_summary = "step completed (no summary)."
        self._last_step_summary = step_summary

        obs = DataCleaningObservation(
            done=self._done,
            reward=immediate,
            preview=self._df.head(8).to_dict(orient="records"),
            column_names=list(self._df.columns),
            issues=self._detect_issues(),
            policy_rules=list(DEFAULT_POLICY_RULES),
            policy_warnings=_detect_policy_warnings(self._df),
            task_name=self._task_name,
            task_difficulty=spec["difficulty"],
            max_steps=max_steps,
            history=list(self._history),
            cumulative_reward=self._cumulative_reward,
            reward_breakdown=EpisodeReward(
                immediate=immediate,
                cumulative=self._cumulative_reward,
                terminal_grader=terminal_grader,
            ),
            terminal_grader_score=terminal_grader,
            instruction=self._instruction_text,
            last_step_summary=step_summary,
        )
        return self._apply_transform(obs)

    def _terminal_observation(self, repeat: bool = False) -> DataCleaningObservation:
        spec = TASK_SPECS[self._task_name]
        imm = -0.01 if repeat else 0.0
        return DataCleaningObservation(
            done=True,
            reward=imm,
            preview=self._df.head(8).to_dict(orient="records"),
            column_names=list(self._df.columns),
            issues=self._detect_issues(),
            policy_rules=list(DEFAULT_POLICY_RULES),
            policy_warnings=_detect_policy_warnings(self._df),
            task_name=self._task_name,
            task_difficulty=spec["difficulty"],
            max_steps=spec["max_steps"],
            history=list(self._history),
            cumulative_reward=self._cumulative_reward,
            reward_breakdown=EpisodeReward(
                immediate=imm,
                cumulative=self._cumulative_reward,
                terminal_grader=None,
            ),
            terminal_grader_score=None,
            instruction=self._instruction_text,
            last_step_summary=self._last_step_summary,
        )

    @property
    def state(self) -> State:
        return State(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_name=self._task_name,
            cumulative_reward=self._cumulative_reward,
            done=self._done,
        )

    def write_working_table_csv(self, path: Path | str) -> Path:
        """Write the current working DataFrame to *path* (baseline / tooling)."""
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        self._df.to_csv(p, index=False)
        return p

    def working_row_count(self) -> int:
        """Number of rows in the working table (for reports)."""
        return int(len(self._df))

    def working_column_names(self) -> List[str]:
        """Column names of the working table."""
        return list(self._df.columns)

    def working_preview_records(self, n: int = 15) -> List[Dict[str, Any]]:
        """First *n* rows as JSON-serializable dicts (for reports/audit)."""
        return self._df.head(n).to_dict(orient="records")

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="DataCleaningEnvironment",
            description=(
                "E-commerce data analyst device: clean order tables (dedupe, impute, outliers), "
                "derive Revenue, aggregate metrics by Category, declare plots (Revenue vs time, category sales), "
                "optional CSV export (AUTODATALAB_CSV_DIR), then submit."
            ),
            version="1.0.0",
        )
