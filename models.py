# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pydantic models for BugTriageEnv.
 
This module defines the typed data structures for the Bug Triage environment:
    - BugObservation : what the agent sees at each step
    - BugAction      : what the agent can do
    - BugState       : internal environment state (returned by state())
    - BugReward      : reward breakdown for transparency
    - BugIssue       : a single synthetic GitHub-style issue
    - TaskConfig     : configuration per task (easy / medium / hard)
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field, BaseModel


class SeverityLabel(str, Enum):
    """Severity classification labels for Task 1 (Easy)."""
    CRITICAL = "critical"   # System down / data loss / security breach
    HIGH     = "high"       # Major feature broken, no workaround
    MEDIUM   = "medium"     # Feature broken but workaround exists
    LOW      = "low"        # Minor issue, cosmetic, or typo
 
 
class TeamLabel(str, Enum):
    """Team routing labels for Task 2 (Medium)."""
    FRONTEND  = "frontend"   # UI, CSS, React, browser issues
    BACKEND   = "backend"    # API, database, server-side logic
    SECURITY  = "security"   # Auth, CVEs, data exposure, injection
    DEVOPS    = "devops"     # CI/CD, deployment, infra, Docker
 
 
class ActionType(str, Enum):
    """All possible action types the agent can take."""
    CLASSIFY       = "classify"        # Task 1: assign a severity label
    ASSIGN_TEAM    = "assign_team"     # Task 2: route to a team
    SUGGEST_FIX    = "suggest_fix"     # Task 3: write a fix suggestion
    SUBMIT         = "submit"          # Mark current issue as done
    SKIP           = "skip"            # Skip current issue (penalty applied)
 
 
class TaskMode(str, Enum):
    """The three task difficulty modes."""
    EASY   = "easy"    # Task 1: classify severity only
    MEDIUM = "medium"  # Task 2: classify + assign team
    HARD   = "hard"    # Task 3: classify + assign + suggest fix
 
 
# ---------------------------------------------------------------------------
# BugIssue — a single GitHub-style bug issue (the raw data)
# ---------------------------------------------------------------------------
 
class BugIssue(BaseModel):
    """
    Represents one synthetic GitHub-style bug issue.
 
    This is the core data unit. Each issue is loaded from
    server/data/bug_issues.json and presented to the agent
    inside a BugObservation.
 
    Fields:
        issue_id        : unique identifier e.g. "BUG-001"
        title           : short one-line summary of the bug
        body            : full description of the bug
        labels          : any existing tags on the issue (may be empty)
        comments        : list of developer/user comments on the issue
        ground_truth_severity : correct severity label (used by grader)
        ground_truth_team     : correct team assignment (used by grader)
        ground_truth_fix_keywords : key concepts a good fix should mention
    """
 
    issue_id: str = Field(..., description="Unique issue identifier e.g. BUG-001")
    title: str = Field(..., description="Short title of the bug")
    body: str = Field(..., description="Full bug description")
    labels: list[str] = Field(default_factory=list, description="Existing issue labels")
    comments: list[str] = Field(default_factory=list, description="Comments on the issue")
 
    # Ground truth — hidden from agent, used only by graders
    ground_truth_severity: SeverityLabel = Field(
        ..., description="Correct severity (used by grader, not shown to agent)"
    )
    ground_truth_team: TeamLabel = Field(
        ..., description="Correct team (used by grader, not shown to agent)"
    )
    ground_truth_fix_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords a correct fix suggestion should contain"
    )
 
 
# ---------------------------------------------------------------------------
# BugObservation — what the agent sees at each step
# ---------------------------------------------------------------------------
 
class BugObservation(BaseModel):
    """
    The observation returned to the agent after reset() or step().
 
    The agent uses this to decide its next action. Critically,
    ground truth fields from BugIssue are NOT exposed here —
    the agent must infer severity/team/fix from the issue content.
 
    Fields:
        issue_id        : ID of the current issue being triaged
        title           : issue title
        body            : issue body text
        labels          : any pre-existing labels
        comments        : comments on the issue
        task_mode       : which task the agent is solving (easy/medium/hard)
        step_count      : how many steps have been taken so far
        issues_remaining: how many issues are left in this episode
        last_reward     : reward received from the last action (0.0 on reset)
        feedback        : human-readable feedback from the last grader
        legal_actions   : which ActionTypes are valid right now
    """
 
    issue_id: str = Field(..., description="Current issue ID")
    title: str = Field(..., description="Issue title")
    body: str = Field(..., description="Issue body")
    labels: list[str] = Field(default_factory=list)
    comments: list[str] = Field(default_factory=list)
 
    task_mode: TaskMode = Field(..., description="Current task difficulty")
    step_count: int = Field(default=0, description="Steps taken so far this episode")
    issues_remaining: int = Field(default=0, description="Issues left to triage")
    last_reward: float = Field(default=0.0, description="Reward from last action")
    feedback: str = Field(default="", description="Grader feedback on last action")

    reward: float = Field(default=0.0, description="Reward from last action")
    done: bool = Field(default=False, description="Whether episode is done")
    
    legal_actions: list[ActionType] = Field(
        default_factory=list,
        description="Valid actions the agent can take right now"
    )
 
 
# ---------------------------------------------------------------------------
# BugAction — what the agent sends to step()
# ---------------------------------------------------------------------------
 
class BugAction(BaseModel):
    """
    The action the agent sends to the environment via step().
 
    Depending on action_type, different fields are required:
        CLASSIFY    → severity must be set
        ASSIGN_TEAM → team must be set
        SUGGEST_FIX → fix_suggestion must be set
        SUBMIT      → no extra fields needed
        SKIP        → no extra fields needed (incurs penalty)
 
    Fields:
        action_type    : which action to perform
        severity       : severity label (required for CLASSIFY)
        team           : team label (required for ASSIGN_TEAM)
        fix_suggestion : free-text fix suggestion (required for SUGGEST_FIX)
    """
 
    action_type: ActionType = Field(..., description="Type of action to perform")
 
    severity: Optional[SeverityLabel] = Field(
        default=None,
        description="Severity classification — required when action_type=CLASSIFY"
    )
    team: Optional[TeamLabel] = Field(
        default=None,
        description="Team assignment — required when action_type=ASSIGN_TEAM"
    )
    fix_suggestion: Optional[str] = Field(
        default=None,
        description="Free-text fix suggestion — required when action_type=SUGGEST_FIX"
    )
 
 
# ---------------------------------------------------------------------------
# BugState — full internal state returned by state()
# ---------------------------------------------------------------------------
 
class BugState(BaseModel):
    """
    The full environment state, returned by the state() API call.
 
    Unlike BugObservation (agent's partial view), BugState contains
    everything including scores, ground truth tracking, and config.
    Used for debugging, evaluation, and the baseline script.
 
    Fields:
        task_mode           : current task difficulty
        episode_issues      : total issues in this episode
        current_issue_index : index of the issue being triaged now
        step_count          : total steps taken this episode
        total_reward        : cumulative reward so far
        scores_per_issue    : per-issue score breakdown
        is_done             : whether the episode has ended
        current_issue_id    : ID of the current issue
    """
 
    task_mode: TaskMode = Field(..., description="Task difficulty mode")
    episode_issues: int = Field(..., description="Total issues in this episode")
    current_issue_index: int = Field(default=0, description="Current issue index")
    step_count: int = Field(default=0, description="Total steps this episode")
    total_reward: float = Field(default=0.0, description="Cumulative reward")
    scores_per_issue: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per-issue score breakdown [{issue_id, severity_score, team_score, fix_score}]"
    )
    is_done: bool = Field(default=False, description="Whether the episode is finished")
    current_issue_id: str = Field(default="", description="ID of current issue")
 
 
# ---------------------------------------------------------------------------
# BugReward — transparent reward breakdown (returned in step() info dict)
# ---------------------------------------------------------------------------
 
class BugReward(BaseModel):
    """
    Detailed reward breakdown for a single step.
 
    Returned inside the info dict of StepResult so the agent
    (and the researcher) can see exactly why it got a certain reward.
 
    Scoring breakdown per task:
        EASY   (max 1.0) → severity_score (1.0)
        MEDIUM (max 1.0) → severity_score (0.5) + team_score (0.5)
        HARD   (max 1.0) → severity_score (0.3) + team_score (0.3) + fix_score (0.4)
 
    Penalties:
        skip_penalty : -0.2 for skipping an issue
        step_penalty : -0.05 per extra step beyond the minimum needed
    """
 
    severity_score: float = Field(default=0.0, description="Score for severity classification (0.0-1.0)")
    team_score: float = Field(default=0.0, description="Score for team assignment (0.0-1.0)")
    fix_score: float = Field(default=0.0, description="Score for fix suggestion quality (0.0-1.0)")
    skip_penalty: float = Field(default=0.0, description="Penalty for skipping (-0.2 if skipped)")
    step_penalty: float = Field(default=0.0, description="Penalty for extra steps (-0.05 each)")
    total: float = Field(default=0.0, description="Final combined reward for this step")
    feedback: str = Field(default="", description="Human-readable explanation of the reward")
 
 
# ---------------------------------------------------------------------------
# TaskConfig — per-task configuration
# ---------------------------------------------------------------------------
 
class TaskConfig(BaseModel):
    """
    Configuration for each task mode.
 
    Defines how many issues the agent must triage per episode,
    and the maximum steps allowed before the episode is force-terminated.
 
    Fields:
        task_mode       : easy / medium / hard
        num_issues      : number of issues per episode
        max_steps       : maximum steps before forced termination
        description     : human-readable task description
    """
 
    task_mode: TaskMode
    num_issues: int = Field(..., description="Issues per episode")
    max_steps: int = Field(..., description="Max steps before termination")
    description: str = Field(..., description="Human-readable task description")
 
 
# ---------------------------------------------------------------------------
# Default task configurations
# ---------------------------------------------------------------------------
 
TASK_CONFIGS: dict[TaskMode, TaskConfig] = {
    TaskMode.EASY: TaskConfig(
        task_mode=TaskMode.EASY,
        num_issues=5,
        max_steps=15,
        description=(
            "Classify the severity of each bug issue as "
            "critical / high / medium / low."
        ),
    ),
    TaskMode.MEDIUM: TaskConfig(
        task_mode=TaskMode.MEDIUM,
        num_issues=5,
        max_steps=20,
        description=(
            "Classify severity AND assign each issue to the correct team: "
            "frontend / backend / security / devops."
        ),
    ),
    TaskMode.HARD: TaskConfig(
        task_mode=TaskMode.HARD,
        num_issues=5,
        max_steps=30,
        description=(
            "Classify severity, assign team, AND write a concrete fix suggestion "
            "that covers the root cause."
        ),
    ),
}
