# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
client.py — Typed WebSocket client for BugTriageEnv.

Confirmed from source inspection:
    StepResult      → openenv.core.client_types.StepResult
    _parse_result() → receives response.get("data", {})
                    → must return StepResult[BugObservation]

StepResult fields (confirmed from source):
    observation : ObsT
    reward      : Optional[float]
    done        : bool = False

Server response structure (confirmed from debug output):
    data = {
        "observation": {
            "observation": {    ← double nested
                "issue_id": ...,
                "title": ...,
                "reward": 0.5,  ← reward embedded IN observation
                "done": False,  ← done embedded IN observation
                ...
            }
        },
        "reward": 0.0,
        "done": False
    }

Since reward/done are embedded in BugObservation, we read them
from the parsed observation object directly.
"""

from __future__ import annotations

from typing import Any

try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient
except ImportError as e:
    raise ImportError(
        "openenv-core is required. Install with: pip install openenv-core"
    ) from e

try:
    from .models import BugAction, BugObservation, BugState
except ImportError:
    from models import BugAction, BugObservation, BugState


# ---------------------------------------------------------------------------
# Helper — unwrap double-nested observation from server response
# ---------------------------------------------------------------------------

def _unwrap_observation(data: dict[str, Any]) -> dict[str, Any]:
    """
    Unwrap the observation from the server response.

    Confirmed server response structure from debug:
        data = {
            "observation": {
                "observation": {    ← actual BugObservation fields here
                    "issue_id": "BUG-001",
                    "title": "...",
                    "reward": 0.5,
                    "done": False,
                    ...
                }
            },
            "reward": 0.0,
            "done": False
        }

    Handles both single and double nesting defensively.
    """
    obs = data.get("observation", {})

    # Double nested — {"observation": {"observation": {actual fields}}}
    if "observation" in obs and isinstance(obs["observation"], dict):
        inner = obs["observation"]
        if "issue_id" in inner:
            return inner

    # Single nested — {"observation": {actual fields}}
    if "issue_id" in obs:
        return obs

    # Fallback — search one level deep for issue_id
    for key, value in data.items():
        if isinstance(value, dict) and "issue_id" in value:
            return value

    # Last resort
    return obs


# ---------------------------------------------------------------------------
# Client class
# ---------------------------------------------------------------------------

class MyEnv(EnvClient[BugObservation, BugAction, BugState]):
    """
    Typed WebSocket client for BugTriageEnv.

    Connects to the BugTriageEnv server over WebSocket and provides
    fully typed reset(), step(), and state() methods.

    Usage:
        # Async context manager (recommended)
        async with MyEnv(base_url="http://localhost:8000") as env:
            result = await env.reset(task_mode="easy", seed=42)
            while not result.done:
                action = BugAction(
                    action_type=ActionType.CLASSIFY,
                    severity=SeverityLabel.HIGH
                )
                result = await env.step(action)
            state = await env.state()
            print(f"Total reward: {state.total_reward}")

        # From Hugging Face Hub
        env = MyEnv.from_hub("your-username/bug-triage-env")
    """

    # -----------------------------------------------------------------------
    # _step_payload() — serialize BugAction → dict for the server
    # -----------------------------------------------------------------------

    def _step_payload(self, action: BugAction) -> dict[str, Any]:
        """
        Convert BugAction to JSON-serializable dict.
        Called internally by step() before sending over WebSocket.
        Excludes None fields to keep payload clean.
        """
        return action.model_dump(exclude_none=True)

    # -----------------------------------------------------------------------
    # _parse_result() — deserialize server response → StepResult[BugObservation]
    #
    # Confirmed:
    #   - Called as: self._parse_result(response.get("data", {}))
    #   - Must return: StepResult[ObsT]
    #   - StepResult fields: observation, reward, done (no info field)
    # -----------------------------------------------------------------------

    def _parse_result(self, data: dict[str, Any]) -> StepResult:
        """
        Parse server response into StepResult[BugObservation].

        Unwraps double-nested observation and reads reward/done
        from the embedded fields in BugObservation.
        """
        # Step 1 — unwrap double nesting to get actual BugObservation fields
        obs_data = _unwrap_observation(data)

        # Step 2 — parse into typed BugObservation
        try:
            observation = BugObservation(**obs_data)
        except Exception as e:
            raise ValueError(
                f"Failed to parse BugObservation from server response.\n"
                f"obs_data keys: {list(obs_data.keys())}\n"
                f"Full data keys: {list(data.keys())}\n"
                f"Error: {e}"
            )

        # Step 3 — build StepResult
        # reward and done are embedded in BugObservation (confirmed from design)
        # Also check top-level data as fallback
        # reward = observation.reward
        reward = data.get("reward")
        if reward is None:
            reward = observation.reward

        done = data.get("done")
        if done is None:
            done = observation.done

        return StepResult(
            observation=observation,
            reward=float(reward) if reward is not None else None,
            done=bool(done),
        )

    # -----------------------------------------------------------------------
    # _parse_state() — deserialize state response → BugState
    # -----------------------------------------------------------------------

    def _parse_state(self, data: dict[str, Any]) -> BugState:
        """
        Parse state() response into typed BugState.

        Handles both flat and nested state responses defensively.
        BugState fields: task_mode, episode_issues, current_issue_index,
                         step_count, total_reward, scores_per_issue,
                         is_done, current_issue_id
        """
        # Try flat — data has BugState fields directly
        if "task_mode" in data:
            state_data = data

        # Try nested under "state"
        elif "state" in data and isinstance(data["state"], dict):
            state_data = data["state"]
            if "task_mode" not in state_data:
                # One more level deep
                for val in state_data.values():
                    if isinstance(val, dict) and "task_mode" in val:
                        state_data = val
                        break

        # Fallback — search for task_mode anywhere one level deep
        else:
            state_data = data
            for key, value in data.items():
                if isinstance(value, dict) and "task_mode" in value:
                    state_data = value
                    break

        try:
            return BugState(**state_data)
        except Exception as e:
            raise ValueError(
                f"Failed to parse BugState from server response.\n"
                f"state_data keys: {list(state_data.keys())}\n"
                f"Error: {e}"
            )