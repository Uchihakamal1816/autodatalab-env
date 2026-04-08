# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed Action, Observation, and reward structures for the data-cleaning OpenEnv."""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Static governance text returned on every observation (production-style constraints).
DEFAULT_POLICY_RULES: List[str] = [
    "Identifier-like columns (OrderID, CustomerID, Name, email, …) are not imputed or dropped here; "
    "define business rules in production.",
    "Imputation targets in each benchmark are named in the task instruction (e.g. Price).",
    "Plots record declared visualizations; some tasks require specific charts (Revenue vs time, etc.).",
]


class EpisodeReward(BaseModel):
    """Structured reward signal (partial progress + terminal grader on [0, 1])."""

    model_config = ConfigDict(extra="forbid")

    immediate: float = Field(..., description="Reward for the last transition")
    cumulative: float = Field(..., description="Sum of immediate rewards this episode")
    terminal_grader: Optional[float] = Field(
        default=None,
        description="Final deterministic grader score in [0, 1] when the episode ends",
    )


class DataCleaningAction(Action):
    """Tabular data cleaning operation (mirrors common pandas workflows)."""

    # LLMs often add extra keys; ignore them instead of failing validation (noop loop).
    model_config = ConfigDict(extra="ignore")

    action_type: Literal[
        "remove_duplicates",
        "fill_missing",
        "drop_column",
        "normalize",
        "remove_outliers",
        "derive_revenue",
        "compute_metrics",
        "compute_kpis",
        "plot",
        "export_csv",
        "submit",
        "noop",
    ] = Field(..., description="Which operation to apply")

    column: Optional[str] = Field(
        default=None,
        description="Target column for column-wise operations",
    )
    method: Optional[Literal["mean", "median", "mode"]] = Field(
        default=None,
        description="Imputation strategy for fill_missing",
    )
    z_threshold: float = Field(
        default=3.0,
        ge=0.5,
        le=10.0,
        description="Z-score cutoff for remove_outliers",
    )

    @field_validator("z_threshold", mode="before")
    @classmethod
    def _z_threshold_drop_null(cls, v: Any) -> Any:
        # LLM JSON often includes "z_threshold": null; treat as unset → default.
        if v is None:
            return 3.0
        return v

    x: Optional[str] = Field(default=None, description="Plot x column (if action_type=plot)")
    y: Optional[str] = Field(default=None, description="Plot y column (if action_type=plot)")
    plot_type: Optional[Literal["scatter", "bar", "histogram"]] = Field(
        default=None,
        description="Plot kind for plot action",
    )
    export_basename: Optional[str] = Field(
        default=None,
        description="Optional safe filename stem for export_csv (ignored if empty)",
    )
    description: Optional[str] = Field(
        default=None,
        description="Short human-readable note: what you did or why (optional; not used for grading)",
    )


class DataCleaningObservation(Observation):
    """What the agent sees: preview, detected issues, task context, and reward breakdown."""

    preview: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="First rows of the working dataframe as JSON-serializable dicts",
    )
    column_names: List[str] = Field(default_factory=list)
    issues: List[str] = Field(
        default_factory=list,
        description="Heuristic issue tags, e.g. duplicates, missing_values",
    )
    policy_rules: List[str] = Field(
        default_factory=lambda: list(DEFAULT_POLICY_RULES),
        description="Static data-governance rules (same every step)",
    )
    policy_warnings: List[str] = Field(
        default_factory=list,
        description="Per-step policy flags, e.g. implausible expiry values (separate from generic issues)",
    )
    task_name: str = Field(default="easy", description="easy | medium | hard")
    task_difficulty: str = Field(
        default="easy",
        description="Human-readable difficulty label",
    )
    max_steps: int = Field(default=40, ge=1)
    history: List[str] = Field(
        default_factory=list,
        description="Serialized actions taken this episode",
    )
    cumulative_reward: float = Field(default=0.0)
    reward_breakdown: Optional[EpisodeReward] = None
    terminal_grader_score: Optional[float] = Field(
        default=None,
        description="0–1 grader when done=True",
    )
    instruction: str = Field(
        default="",
        description="Short task description for the agent",
    )
    last_step_summary: str = Field(
        default="",
        description="Authoritative effect of the previous step (from the environment, not the LLM)",
    )
