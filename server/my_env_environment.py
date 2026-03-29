# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
My Env Environment Implementation.

Core environment logic for BugTriageEnv.
 
This module implements the OpenEnv-compatible environment class that:
    1. Loads bug issues from server/data/bug_issues.json
    2. Exposes reset() / step() / state() APIs
    3. Runs 3 task graders (easy / medium / hard)
    4. Computes meaningful partial-progress reward signals
 
Class:
    MyEnvironment — main environment class registered with OpenEnv
 
Task Graders:
    _grade_easy()   — severity classification (0.0–1.0)
    _grade_medium() — severity + team routing (0.0–1.0)
    _grade_hard()   — severity + team + fix suggestion (0.0–1.0)
 
Reward Structure:
    EASY   : severity_score (1.0 max)
    MEDIUM : severity_score (0.5) + team_score (0.5)
    HARD   : severity_score (0.3) + team_score (0.3) + fix_score (0.4)
    Penalty: -0.2 for skip, -0.05 per extra step

"""
from __future__ import annotations

import json
import os
import random
from typing import Any,Optional

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State


# try:
#     from ..models import MyAction, MyObservation
# except ImportError:
#     from models import MyAction, MyObservation


from openenv.core.env_server.interfaces import Environment as BaseEnvironment
from openenv.core.env_server.types import EnvironmentMetadata
 
# ---------------------------------------------------------------------------
# Our Pydantic models
# ---------------------------------------------------------------------------
from models import (
    ActionType,
    BugAction,
    BugIssue,
    BugObservation,
    BugReward,
    BugState,
    SeverityLabel,
    TaskMode,
    TASK_CONFIGS,
)
 
 
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
 
_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "bug_issues.json")
 
_FIX_KEYWORD_THRESHOLD = 0.4
 
_SEVERITY_PARTIAL = {
    SeverityLabel.CRITICAL: {
        SeverityLabel.CRITICAL: 1.0,
        SeverityLabel.HIGH:     0.5,
        SeverityLabel.MEDIUM:   0.2,
        SeverityLabel.LOW:      0.0,
    },
    SeverityLabel.HIGH: {
        SeverityLabel.CRITICAL: 0.5,
        SeverityLabel.HIGH:     1.0,
        SeverityLabel.MEDIUM:   0.5,
        SeverityLabel.LOW:      0.1,
    },
    SeverityLabel.MEDIUM: {
        SeverityLabel.CRITICAL: 0.2,
        SeverityLabel.HIGH:     0.5,
        SeverityLabel.MEDIUM:   1.0,
        SeverityLabel.LOW:      0.5,
    },
    SeverityLabel.LOW: {
        SeverityLabel.CRITICAL: 0.0,
        SeverityLabel.HIGH:     0.1,
        SeverityLabel.MEDIUM:   0.5,
        SeverityLabel.LOW:      1.0,
    },
}
 
 
# ---------------------------------------------------------------------------
# Helper — load issues from JSON
# ---------------------------------------------------------------------------
 
def _load_issues(path: str = _DATA_PATH) -> list[BugIssue]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: python generate_data.py"
        )
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [BugIssue(**item) for item in raw]
 
 
# ---------------------------------------------------------------------------
# Main Environment Class
# ---------------------------------------------------------------------------
 
class MyEnvironment(BaseEnvironment):
    """
    BugTriageEnv — AI Bug Triage & Response Environment.
 
    Inherits from openenv.core.env_server.interfaces.Environment.
 
    Key design decisions based on source inspection:
        - reset() returns BugObservation DIRECTLY
        - step() returns BugObservation DIRECTLY
        - reward and done are embedded IN BugObservation
        - state is a @property (not a method)
        - The HTTP server extracts reward/done from the observation
    """
 
    def __init__(self) -> None:
        super().__init__()
 
        # Load dataset once at startup
        self._all_issues: list[BugIssue] = _load_issues()
 
        # Episode state
        self._task_mode: TaskMode = TaskMode.EASY
        self._episode_issues: list[BugIssue] = []
        self._current_index: int = 0
        self._step_count: int = 0
        self._total_reward: float = 0.0
        self._scores_per_issue: list[dict[str, Any]] = []
        self._is_done: bool = False
 
        # Per-issue action accumulators
        self._current_severity: SeverityLabel | None = None
        self._current_team: str | None = None
        self._current_fix: str | None = None
        self._last_feedback: str = ""
        self._last_reward: float = 0.0
 
    # -----------------------------------------------------------------------
    # get_metadata() — regular method
    # -----------------------------------------------------------------------
 
    def get_metadata(self) -> EnvironmentMetadata:
        """Return environment metadata for the /schema endpoint."""
        return EnvironmentMetadata(
            name="bug-triage-env",
            version="1.0.0",
            description=(
                "BugTriageEnv: AI agent triages software bug reports. "
                "Tasks: severity classification (easy), "
                "team routing (medium), fix suggestion (hard)."
            ),
        )
 
    # -----------------------------------------------------------------------
    # reset() — returns BugObservation DIRECTLY
    # Confirmed signature: reset(seed, episode_id, **kwargs) -> ObsT
    # -----------------------------------------------------------------------
 
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> BugObservation:
        """
        Reset for a new episode.
 
        Returns BugObservation directly with reward=0.0, done=False embedded.
 
        kwargs (from reset request options):
            task_mode : "easy" | "medium" | "hard"  (default: "easy")
        """
        # Reset rubric state if one is attached
        self._reset_rubric()
 
        # Parse task_mode from kwargs
        mode_str = kwargs.get("task_mode", "easy")
        try:
            self._task_mode = TaskMode(mode_str)
        except ValueError:
            self._task_mode = TaskMode.EASY
 
        # Apply seed
        if seed is not None:
            random.seed(seed)
 
        # Sample episode issues
        config = TASK_CONFIGS[self._task_mode]
        self._episode_issues = random.sample(
            self._all_issues,
            min(config.num_issues, len(self._all_issues))
        )
 
        # Reset all episode state
        self._current_index = 0
        self._step_count = 0
        self._total_reward = 0.0
        self._scores_per_issue = []
        self._is_done = False
        self._last_feedback = "Episode started. Triage the first issue."
        self._last_reward = 0.0
        self._reset_issue_accumulators()
 
        # Build observation with reward/done embedded
        obs = self._build_observation()
        obs.reward = 0.0
        obs.done = False
        return obs
 
    # -----------------------------------------------------------------------
    # step() — returns BugObservation DIRECTLY
    # Confirmed signature: step(action, timeout_s, **kwargs) -> ObsT
    # -----------------------------------------------------------------------
 
    def step(
        self,
        action: Any,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> BugObservation:
        """
        Execute one agent action.
 
        Returns BugObservation directly with reward and done embedded.
        The OpenEnv server extracts reward/done from the observation.
 
        action: BugAction instance or dict with BugAction fields.
        """
        # Episode already done
        if self._is_done:
            obs = self._build_observation()
            obs.reward = 0.0
            obs.done = True
            return obs
 
        # Parse action — may arrive as dict or BugAction
        try:
            if isinstance(action, dict):
                bug_action = BugAction(**action)
            elif isinstance(action, BugAction):
                bug_action = action
            else:
                bug_action = BugAction(**dict(action))
        except Exception as e:
            self._last_feedback = f"Invalid action format: {e}"
            self._last_reward = -0.1
            obs = self._build_observation()
            obs.reward = -0.1
            obs.done = False
            return obs
 
        self._step_count += 1
        reward = 0.0
        current_issue = self._episode_issues[self._current_index]
 
        # ----------------------------------------------------------------
        # Dispatch action type
        # ----------------------------------------------------------------
 
        if bug_action.action_type == ActionType.SKIP:
            reward = -0.2
            self._last_feedback = (
                f"Skipped {current_issue.issue_id}. Penalty: -0.2. "
                f"Truth: severity={current_issue.ground_truth_severity.value}, "
                f"team={current_issue.ground_truth_team.value}."
            )
            self._scores_per_issue.append({
                "issue_id": current_issue.issue_id,
                "severity_score": 0.0,
                "team_score": 0.0,
                "fix_score": 0.0,
                "total": reward,
                "skipped": True,
            })
            self._advance_to_next_issue()
 
        elif bug_action.action_type == ActionType.CLASSIFY:
            if bug_action.severity is None:
                reward = -0.1
                self._last_feedback = "CLASSIFY requires 'severity' field."
            else:
                self._current_severity = bug_action.severity
                self._last_feedback = (
                    f"Severity recorded: '{bug_action.severity.value}'. "
                    + self._next_action_hint()
                )
 
        elif bug_action.action_type == ActionType.ASSIGN_TEAM:
            if bug_action.team is None:
                reward = -0.1
                self._last_feedback = "ASSIGN_TEAM requires 'team' field."
            else:
                self._current_team = bug_action.team.value
                self._last_feedback = (
                    f"Team recorded: '{bug_action.team.value}'. "
                    + self._next_action_hint()
                )
 
        elif bug_action.action_type == ActionType.SUGGEST_FIX:
            if not bug_action.fix_suggestion or len(bug_action.fix_suggestion.strip()) < 10:
                reward = -0.1
                self._last_feedback = "SUGGEST_FIX requires fix_suggestion (min 10 chars)."
            else:
                self._current_fix = bug_action.fix_suggestion
                self._last_feedback = "Fix recorded. Call SUBMIT to finalize."
 
        elif bug_action.action_type == ActionType.SUBMIT:
            bug_reward = self._grade_current_issue(current_issue)
            reward = bug_reward.total
            self._last_feedback = bug_reward.feedback
            self._scores_per_issue.append({
                "issue_id": current_issue.issue_id,
                "severity_score": bug_reward.severity_score,
                "team_score": bug_reward.team_score,
                "fix_score": bug_reward.fix_score,
                "total": bug_reward.total,
                "skipped": False,
            })
            self._advance_to_next_issue()
 
        # Step penalty for inefficiency
        if bug_action.action_type in (ActionType.SUBMIT, ActionType.SKIP):
            min_steps = self._min_steps_for_task()
            if self._step_count > min_steps * len(self._episode_issues):
                extra = self._step_count - min_steps * len(self._episode_issues)
                step_pen = round(min(extra * 0.05, 0.2), 2)
                reward -= step_pen
                self._last_feedback += f" Step penalty: -{step_pen}."
 
        self._total_reward += reward
        self._last_reward = round(reward, 4)
 
        # Check max steps termination
        config = TASK_CONFIGS[self._task_mode]
        if not self._is_done and self._step_count >= config.max_steps:
            self._is_done = True
            self._last_feedback += " Max steps reached. Episode terminated."
 
        # Build observation and embed reward/done
        obs = self._build_observation()
        obs.reward = self._last_reward
        obs.done = self._is_done
        return obs
 
    # -----------------------------------------------------------------------
    # state — PROPERTY (confirmed: @property @abstractmethod in base class)
    # -----------------------------------------------------------------------
 
    @property
    def state(self) -> dict[str, Any]:
        """
        Return full environment state as dict.
 
        Must be a @property — confirmed from Environment source:
            @property
            @abstractmethod
            def state(self) -> StateT: ...
        """
        current_id = ""
        if not self._is_done and self._current_index < len(self._episode_issues):
            current_id = self._episode_issues[self._current_index].issue_id
 
        return BugState(
            task_mode=self._task_mode,
            episode_issues=len(self._episode_issues),
            current_issue_index=self._current_index,
            step_count=self._step_count,
            total_reward=round(self._total_reward, 4),
            scores_per_issue=self._scores_per_issue,
            is_done=self._is_done,
            current_issue_id=current_id,
        ).model_dump()
 
    # -----------------------------------------------------------------------
    # close() — cleanup
    # -----------------------------------------------------------------------
 
    def close(self) -> None:
        """Clean up resources."""
        pass
 
    # -----------------------------------------------------------------------
    # Graders
    # -----------------------------------------------------------------------
 
    def _grade_current_issue(self, issue: BugIssue) -> BugReward:
        if self._task_mode == TaskMode.EASY:
            return self._grade_easy(issue)
        elif self._task_mode == TaskMode.MEDIUM:
            return self._grade_medium(issue)
        else:
            return self._grade_hard(issue)
 
    def _grade_easy(self, issue: BugIssue) -> BugReward:
        """Task 1: severity classification only."""
        severity_score = 0.0
        feedback_parts = []
 
        if self._current_severity is None:
            feedback_parts.append("No severity set before SUBMIT (score=0).")
        else:
            severity_score = _SEVERITY_PARTIAL[issue.ground_truth_severity].get(
                self._current_severity, 0.0
            )
            if severity_score == 1.0:
                feedback_parts.append(
                    f"✓ Severity correct: {self._current_severity.value}."
                )
            else:
                feedback_parts.append(
                    f"✗ Severity: predicted={self._current_severity.value}, "
                    f"truth={issue.ground_truth_severity.value} "
                    f"(score={severity_score})."
                )
 
        return BugReward(
            severity_score=severity_score,
            total=round(severity_score, 4),
            feedback=" ".join(feedback_parts),
        )
 
    def _grade_medium(self, issue: BugIssue) -> BugReward:
        """Task 2: severity + team routing."""
        severity_score = 0.0
        if self._current_severity is not None:
            severity_score = _SEVERITY_PARTIAL[issue.ground_truth_severity].get(
                self._current_severity, 0.0
            )
 
        team_score = 0.0
        if self._current_team is not None:
            team_score = (
                1.0 if self._current_team == issue.ground_truth_team.value else 0.0
            )
 
        total = round(severity_score * 0.5 + team_score * 0.5, 4)
        sev_pred = self._current_severity.value if self._current_severity else "none"
        team_pred = self._current_team or "none"
 
        feedback = (
            f"{'✓' if severity_score == 1.0 else '✗'} "
            f"Severity: {sev_pred} "
            f"(truth={issue.ground_truth_severity.value}, score={severity_score}). "
            f"{'✓' if team_score == 1.0 else '✗'} "
            f"Team: {team_pred} "
            f"(truth={issue.ground_truth_team.value}, score={team_score}). "
            f"Total: {total}/1.0."
        )
 
        return BugReward(
            severity_score=severity_score,
            team_score=team_score,
            total=total,
            feedback=feedback,
        )
 
    def _grade_hard(self, issue: BugIssue) -> BugReward:
        """Task 3: severity + team + fix suggestion."""
        severity_score = 0.0
        if self._current_severity is not None:
            severity_score = _SEVERITY_PARTIAL[issue.ground_truth_severity].get(
                self._current_severity, 0.0
            )
 
        team_score = 0.0
        if self._current_team is not None:
            team_score = (
                1.0 if self._current_team == issue.ground_truth_team.value else 0.0
            )
 
        fix_score = 0.0
        matched: list[str] = []
        if self._current_fix and issue.ground_truth_fix_keywords:
            fix_lower = self._current_fix.lower()
            matched = [
                kw for kw in issue.ground_truth_fix_keywords
                if kw.lower() in fix_lower
            ]
            ratio = len(matched) / len(issue.ground_truth_fix_keywords)
            fix_score = round(min(ratio / _FIX_KEYWORD_THRESHOLD, 1.0), 4)
 
        total = round(
            severity_score * 0.3 + team_score * 0.3 + fix_score * 0.4, 4
        )
 
        sev_pred = self._current_severity.value if self._current_severity else "none"
        team_pred = self._current_team or "none"
 
        feedback = (
            f"Severity: {sev_pred} "
            f"(truth={issue.ground_truth_severity.value}, score={severity_score}). "
            f"Team: {team_pred} "
            f"(truth={issue.ground_truth_team.value}, score={team_score}). "
            f"Fix: {len(matched)}/{len(issue.ground_truth_fix_keywords)} "
            f"keywords matched (score={fix_score}). "
            f"Total: {total}/1.0."
        )
 
        return BugReward(
            severity_score=severity_score,
            team_score=team_score,
            fix_score=fix_score,
            total=total,
            feedback=feedback,
        )
 
    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------
 
    def _build_observation(self) -> BugObservation:
        """Build BugObservation for the current issue."""
        if self._is_done or self._current_index >= len(self._episode_issues):
            return BugObservation(
                issue_id="DONE",
                title="Episode complete.",
                body="All issues triaged. Call reset() to start a new episode.",
                task_mode=self._task_mode,
                step_count=self._step_count,
                issues_remaining=0,
                last_reward=self._last_reward,
                feedback=self._last_feedback,
                legal_actions=[],
                reward=self._last_reward,
                done=True,
            )
 
        issue = self._episode_issues[self._current_index]
        return BugObservation(
            issue_id=issue.issue_id,
            title=issue.title,
            body=issue.body,
            labels=issue.labels,
            comments=issue.comments,
            task_mode=self._task_mode,
            step_count=self._step_count,
            issues_remaining=len(self._episode_issues) - self._current_index,
            last_reward=self._last_reward,
            feedback=self._last_feedback,
            legal_actions=self._get_legal_actions(),
            reward=self._last_reward,
            done=self._is_done,
        )
 
    def _get_legal_actions(self) -> list[ActionType]:
        """Return valid actions for the current task and state."""
        base = [ActionType.SKIP, ActionType.SUBMIT]
        if self._task_mode == TaskMode.EASY:
            return [ActionType.CLASSIFY] + base
        elif self._task_mode == TaskMode.MEDIUM:
            return [ActionType.CLASSIFY, ActionType.ASSIGN_TEAM] + base
        else:
            return [
                ActionType.CLASSIFY,
                ActionType.ASSIGN_TEAM,
                ActionType.SUGGEST_FIX,
            ] + base
 
    def _advance_to_next_issue(self) -> None:
        """Move to next issue or mark episode done."""
        self._current_index += 1
        self._reset_issue_accumulators()
        if self._current_index >= len(self._episode_issues):
            self._is_done = True
 
    def _reset_issue_accumulators(self) -> None:
        """Clear per-issue action state."""
        self._current_severity = None
        self._current_team = None
        self._current_fix = None
 
    def _next_action_hint(self) -> str:
        """Give the agent a hint about what to do next."""
        if self._task_mode == TaskMode.EASY:
            return "Now call SUBMIT."
        elif self._task_mode == TaskMode.MEDIUM:
            missing = []
            if self._current_severity is None:
                missing.append("CLASSIFY")
            if self._current_team is None:
                missing.append("ASSIGN_TEAM")
            return f"Still needed: {', '.join(missing)}." if missing else "Now call SUBMIT."
        else:
            missing = []
            if self._current_severity is None:
                missing.append("CLASSIFY")
            if self._current_team is None:
                missing.append("ASSIGN_TEAM")
            if self._current_fix is None:
                missing.append("SUGGEST_FIX")
            return f"Still needed: {', '.join(missing)}." if missing else "Now call SUBMIT."
 
    def _min_steps_for_task(self) -> int:
        """Minimum steps per issue for the current task."""
        return {
            TaskMode.EASY:   2,
            TaskMode.MEDIUM: 3,
            TaskMode.HARD:   4,
        }[self._task_mode]
 