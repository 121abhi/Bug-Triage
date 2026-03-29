from __future__ import annotations
 
import argparse
import json
import os
import sys
import time
from typing import Any

from dotenv import load_dotenv
load_dotenv()  
 
# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------
api_key = os.environ.get("GEMINI_API_KEY")
try:
    from openai import OpenAI
except ImportError:
    print("❌ openai package required.")
    print("   Run: pip install openai")
    sys.exit(1)
 
try:
    from my_env.client import MyEnv
    from my_env.models import (
        ActionType,
        BugAction,
        BugObservation,
        SeverityLabel,
        TaskMode,
        TeamLabel,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from client import MyEnv
    from models import (
        ActionType,
        BugAction,
        BugObservation,
        SeverityLabel,
        TaskMode,
        TeamLabel,
    )
 
 
# ---------------------------------------------------------------------------
# Gemini configuration
# ---------------------------------------------------------------------------
 
# Gemini's OpenAI-compatible base URL
# Source: https://ai.google.dev/gemini-api/docs/openai
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
 
# Default model — gemini-2.0-flash is fast, capable, and free-tier friendly
# Other options: gemini-2.5-flash, gemini-2.5-pro, gemini-1.5-flash
DEFAULT_MODEL = "gemini-2.5-flash"
 
DEFAULT_SERVER = "http://localhost:8000"
DEFAULT_SEED = 42
 
SEVERITY_VALUES = [s.value for s in SeverityLabel]
TEAM_VALUES = [t.value for t in TeamLabel]
 
 
# ---------------------------------------------------------------------------
# LLM Agent (Gemini-powered)
# ---------------------------------------------------------------------------
 
class GeminiAgent:
    """
    Gemini-powered agent for BugTriageEnv.
 
    Uses the Gemini API via the OpenAI-compatible endpoint.
    This means we use the standard `openai` Python library —
    just pointed at Google's servers instead of OpenAI's.
 
    Key config:
        api_key  → your GEMINI_API_KEY from aistudio.google.com
        base_url → https://generativelanguage.googleapis.com/v1beta/openai/
        model    → any Gemini model name (e.g. gemini-2.0-flash)
    """
 
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        # Point the OpenAI client at Gemini's compatible endpoint
        self.client = OpenAI(
            api_key=api_key,
            base_url=GEMINI_BASE_URL,
        )
        self.model = model
        print(f"  🤖 Agent: Gemini via OpenAI-compatible API")
        print(f"  📡 Model: {self.model}")
        print(f"  🔗 Endpoint: {GEMINI_BASE_URL}")
 
    def decide(self, obs: BugObservation) -> list[BugAction]:
        """
        Given a BugObservation, return the full action sequence for this issue.
 
        Returns actions in the correct order for the task mode:
            EASY   -> [CLASSIFY, SUBMIT]
            MEDIUM -> [CLASSIFY, ASSIGN_TEAM, SUBMIT]
            HARD   -> [CLASSIFY, ASSIGN_TEAM, SUGGEST_FIX, SUBMIT]
        """
        prompt = self._build_prompt(obs)
        response = self._call_gemini(prompt)
        return self._build_actions(response, obs.task_mode)
 
    def _build_prompt(self, obs: BugObservation) -> str:
        """Build the triage prompt from the observation."""
        comments_text = (
            "\n".join(f"  - {c}" for c in obs.comments)
            if obs.comments else "  (none)"
        )
        labels_text = ", ".join(obs.labels) if obs.labels else "(none)"
 
        prompt = f"""You are an expert software engineering triage assistant.
 
Analyze the following bug report carefully and provide your assessment.
 
---
Issue ID : {obs.issue_id}
Title    : {obs.title}
Labels   : {labels_text}
 
Description:
{obs.body}
 
Comments:
{comments_text}
---
 
"""
 
        if obs.task_mode == TaskMode.EASY:
            prompt += f"""Task: SEVERITY CLASSIFICATION
 
Classify the severity of this bug issue.
Choose exactly one severity from: {SEVERITY_VALUES}
 
Severity definitions:
  critical -> System down, data loss, security breach, revenue blocked
  high     -> Major feature broken, no workaround available
  medium   -> Feature broken but a workaround exists
  low      -> Minor issue, cosmetic problem, or typo
 
Respond ONLY with valid JSON, no markdown fences, no explanation:
{{
  "severity": "<one of {SEVERITY_VALUES}>"
}}"""
 
        elif obs.task_mode == TaskMode.MEDIUM:
            prompt += f"""Task: SEVERITY + TEAM ROUTING
 
1. Classify the severity of this bug.
2. Assign it to the correct engineering team.
 
Valid severity values: {SEVERITY_VALUES}
  critical -> System down, data loss, security breach, revenue blocked
  high     -> Major feature broken, no workaround
  medium   -> Feature broken but workaround exists
  low      -> Minor / cosmetic issue
 
Valid team values: {TEAM_VALUES}
  frontend -> UI, CSS, React, browser rendering, JavaScript issues
  backend  -> API, database, server-side logic, queues, workers
  security -> Auth, CVEs, data exposure, injection attacks, tokens
  devops   -> CI/CD, Docker, Kubernetes, deployment, infra, pipelines
 
Respond ONLY with valid JSON, no markdown fences, no explanation:
{{
  "severity": "<one of {SEVERITY_VALUES}>",
  "team": "<one of {TEAM_VALUES}>"
}}"""
 
        else:  # HARD
            prompt += f"""Task: FULL TRIAGE - SEVERITY + TEAM + FIX SUGGESTION
 
1. Classify the severity of this bug.
2. Assign it to the correct engineering team.
3. Write a detailed, technical fix suggestion addressing the root cause.
 
Valid severity values: {SEVERITY_VALUES}
  critical -> System down, data loss, security breach, revenue blocked
  high     -> Major feature broken, no workaround
  medium   -> Feature broken but workaround exists
  low      -> Minor / cosmetic issue
 
Valid team values: {TEAM_VALUES}
  frontend -> UI, CSS, React, browser rendering, JavaScript issues
  backend  -> API, database, server-side logic, queues, workers
  security -> Auth, CVEs, data exposure, injection attacks, tokens
  devops   -> CI/CD, Docker, Kubernetes, deployment, infra, pipelines
 
Fix suggestion guidelines:
  - Be specific and technical (minimum 2 sentences)
  - Identify the root cause clearly
  - Suggest concrete code-level or config-level remediation steps
  - Mention relevant tools, patterns, or best practices
 
Respond ONLY with valid JSON, no markdown fences, no explanation:
{{
  "severity": "<one of {SEVERITY_VALUES}>",
  "team": "<one of {TEAM_VALUES}>",
  "fix_suggestion": "<your detailed technical fix suggestion>"
}}"""
 
        return prompt
 
    def _call_gemini(self, prompt: str) -> dict[str, Any]:
        """
        Call Gemini using the OpenAI-compatible endpoint.
 
        Identical to a regular OpenAI call — only base_url differs.
        Source: https://ai.google.dev/gemini-api/docs/openai
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert software engineering triage assistant. "
                            "Always respond with valid JSON only. "
                            "No markdown code fences. No explanation. Just JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=512,
            )
 
            content = response.choices[0].message.content.strip()
 
            # Strip markdown fences if Gemini adds them anyway
            if "```" in content:
                parts = content.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    try:
                        return json.loads(part)
                    except json.JSONDecodeError:
                        continue
 
            return json.loads(content)
 
        except json.JSONDecodeError as e:
            print(f"  Warning: Gemini returned non-JSON: {e}. Using safe defaults.")
            return {}
        except Exception as e:
            print(f"  Warning: Gemini API call failed: {e}. Using safe defaults.")
            return {}
 
    def _build_actions(
        self, response: dict[str, Any], task_mode: TaskMode
    ) -> list[BugAction]:
        """Convert Gemini JSON response into the BugAction sequence."""
        actions: list[BugAction] = []
 
        # --- CLASSIFY ---
        severity_str = response.get("severity", "medium")
        try:
            severity = SeverityLabel(severity_str)
        except ValueError:
            print(f"  Warning: Invalid severity '{severity_str}', defaulting to 'medium'")
            severity = SeverityLabel.MEDIUM
        actions.append(BugAction(
            action_type=ActionType.CLASSIFY,
            severity=severity,
        ))
 
        if task_mode in (TaskMode.MEDIUM, TaskMode.HARD):
            # --- ASSIGN_TEAM ---
            team_str = response.get("team", "backend")
            try:
                team = TeamLabel(team_str)
            except ValueError:
                print(f"  Warning: Invalid team '{team_str}', defaulting to 'backend'")
                team = TeamLabel.BACKEND
            actions.append(BugAction(
                action_type=ActionType.ASSIGN_TEAM,
                team=team,
            ))
 
        if task_mode == TaskMode.HARD:
            # --- SUGGEST_FIX ---
            fix = response.get(
                "fix_suggestion",
                "Investigate the root cause and apply an appropriate fix based on the error logs."
            )
            if len(fix.strip()) < 10:
                fix = "Investigate the root cause and apply an appropriate fix based on the error logs."
            actions.append(BugAction(
                action_type=ActionType.SUGGEST_FIX,
                fix_suggestion=fix,
            ))
 
        # --- SUBMIT (always last) ---
        actions.append(BugAction(action_type=ActionType.SUBMIT))
 
        return actions
 
 
# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------
 
def run_task(
    agent: GeminiAgent,
    server_url: str,
    task_mode: str,
    seed: int,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run one full episode for a given task and return results."""
    print(f"\n{'='*60}")
    print(f"  TASK: {task_mode.upper()}")
    print(f"{'='*60}")
 
    with MyEnv(base_url=server_url) as env:
        result = env.reset({"task_mode": task_mode, "seed": seed})
        issues_triaged = 0
        start_time = time.time()
 
        while not result.done:
            obs: BugObservation = result.observation
 
            if obs.issue_id == "DONE":
                break
 
            if verbose:
                print(f"\n  Issue {obs.issue_id}: {obs.title[:55]}...")
                print(f"     Remaining: {obs.issues_remaining} | Step: {obs.step_count}")
 
            # Agent decides full action sequence for this issue
            actions = agent.decide(obs)
 
            # Execute actions
            for action in actions:
                result = env.step(action)
                if verbose and action.action_type == ActionType.SUBMIT:
                    print(f"     Reward : {result.reward:.3f}")
                    print(f"     Feedback: {result.observation.feedback[:75]}...")
 
            issues_triaged += 1
 
        state = env.state()
        elapsed = time.time() - start_time
 
        print(f"\n  Done in {elapsed:.1f}s")
        print(f"  Total reward : {state.total_reward:.4f}")
        print(f"  Issues done  : {issues_triaged}/{state.episode_issues}")
        print(f"  Total steps  : {state.step_count}")
 
        return {
            "task": task_mode,
            "model": agent.model,
            "total_reward": state.total_reward,
            "avg_reward": round(state.total_reward / max(issues_triaged, 1), 4),
            "issues_triaged": issues_triaged,
            "episode_issues": state.episode_issues,
            "total_steps": state.step_count,
            "per_issue_scores": state.scores_per_issue,
            "elapsed_seconds": round(elapsed, 2),
        }
 
 
def print_summary(results: list[dict[str, Any]]) -> None:
    """Print a clean summary table."""
    print(f"\n{'='*60}")
    print("  BASELINE RESULTS - Gemini Agent")
    print(f"{'='*60}")
    print(f"  {'Task':<10} {'Avg Score':<12} {'Total':<10} {'Issues':<10} {'Steps'}")
    print(f"  {'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*8}")
    for r in results:
        issues = f"{r['issues_triaged']}/{r['episode_issues']}"
        print(
            f"  {r['task']:<10} "
            f"{r['avg_reward']:<12.4f} "
            f"{r['total_reward']:<10.4f} "
            f"{issues:<10} "
            f"{r['total_steps']}"
        )
    print(f"{'='*60}\n")
 
 
# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
 
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Gemini baseline agent against BugTriageEnv"
    )
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Which task to run (default: all)",
    )
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER,
        help=f"BugTriageEnv server URL (default: {DEFAULT_SERVER})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step output",
    )
    args = parser.parse_args()
 
    # Check for Gemini API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n  GEMINI_API_KEY not set.")
        print()
        print("  How to get a FREE Gemini API key:")
        print("  1. Go to https://aistudio.google.com/apikey")
        print("  2. Click 'Create API Key'")
        print("  3. Copy the key and set it:")
        print()
        print("  Windows PowerShell:")
        print('    $env:GEMINI_API_KEY="your-key-here"')
        print()
        print("  Linux / Mac:")
        print("    export GEMINI_API_KEY=your-key-here")
        sys.exit(1)
 
    print(f"\nBugTriageEnv Baseline - Gemini Agent")
    print(f"   Model   : {args.model}")
    print(f"   Endpoint: {GEMINI_BASE_URL}")
    print(f"   Server  : {args.server}")
    print(f"   Seed    : {args.seed}")
    print(f"   Task    : {args.task}")
 
    agent = GeminiAgent(api_key=api_key, model=args.model)
 
    tasks = (
        ["easy", "medium", "hard"]
        if args.task == "all"
        else [args.task]
    )
 
    results = []
    for task in tasks:
        result = run_task(
            agent=agent,
            server_url=args.server,
            task_mode=task,
            seed=args.seed,
            verbose=not args.quiet,
        )
        results.append(result)
 
    print_summary(results)
 
    output_path = "baseline_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")
 
 
if __name__ == "__main__":
    main()
 