# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""WebSocket client for the Data Cleaning environment."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import (
    DEFAULT_POLICY_RULES,
    DataCleaningAction,
    DataCleaningObservation,
    EpisodeReward,
)


class DataCleaningEnv(EnvClient[DataCleaningAction, DataCleaningObservation, State]):
    """Client for the tabular data-cleaning OpenEnv server."""

    def _step_payload(self, action: DataCleaningAction) -> Dict[str, Any]:
        d = action.model_dump(exclude_none=True)
        # Flatten for wire format
        return d

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[DataCleaningObservation]:
        obs_data = payload.get("observation", {})
        rb = obs_data.get("reward_breakdown")
        reward_breakdown = EpisodeReward.model_validate(rb) if rb else None

        pr = obs_data.get("policy_rules")
        observation = DataCleaningObservation(
            preview=obs_data.get("preview", []),
            column_names=obs_data.get("column_names", []),
            issues=obs_data.get("issues", []),
            policy_rules=list(DEFAULT_POLICY_RULES) if pr is None else pr,
            policy_warnings=obs_data.get("policy_warnings") or [],
            task_name=obs_data.get("task_name", "easy"),
            task_difficulty=obs_data.get("task_difficulty", "easy"),
            max_steps=obs_data.get("max_steps", 40),
            history=obs_data.get("history", []),
            cumulative_reward=float(obs_data.get("cumulative_reward", 0.0)),
            reward_breakdown=reward_breakdown,
            terminal_grader_score=obs_data.get("terminal_grader_score"),
            instruction=obs_data.get("instruction", ""),
            last_step_summary=obs_data.get("last_step_summary", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_name=payload.get("task_name"),
            cumulative_reward=payload.get("cumulative_reward"),
            done=payload.get("done"),
        )
