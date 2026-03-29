"""
Inference Script — BugTriageEnv
===================================
MANDATORY requirements met:
    - Named inference.py and placed in root directory
    - Uses API_BASE_URL, MODEL_NAME, HF_TOKEN environment variables
    - Uses OpenAI Client for all LLM calls
    - Must complete in under 20 minutes
    - Runs on vcpu=2, memory=8gb

Environment variables required:
    API_BASE_URL   The API endpoint for the LLM
                   e.g. https://router.huggingface.co/v1
    MODEL_NAME     The model identifier
                   e.g. Qwen/Qwen2.5-72B-Instruct
    HF_TOKEN       Your Hugging Face API token

Usage:
    # Set environment variables
    export API_BASE_URL=https://router.huggingface.co/v1
    export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
    export HF_TOKEN=hf_your_token_here

    # Run all tasks
    python inference.py

    # Run specific task
    python inference.py --task easy
    python inference.py --task medium
    python inference.py --task hard

    # Run against HF Space
    python inference.py --server https://your-username-bug-triage-env.hf.space
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import textwrap
from typing import Any

from dotenv import load_dotenv

# Load .env file for local development
load_dotenv()

# ---------------------------------------------------------------------------
# MANDATORY environment variables (exactly as specified in submission rules)
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# ---------------------------------------------------------------------------
# Inference configuration
# ---------------------------------------------------------------------------
DEFAULT_SERVER = "http://localhost:8000"
DEFAULT_SEED   = 42
MAX_STEPS      = 50
TEMPERATURE    = 0.1
MAX_TOKENS     = 512
DEBUG          = False

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI
except ImportError:
    print("❌ openai package required. Run: pip install openai")
    sys.exit(1)

try:
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
except ImportError as e:
    print(f"❌ Failed to import BugTriageEnv modules: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Valid label vocabularies
# ---------------------------------------------------------------------------
SEVERITY_VALUES = [s.value for s in SeverityLabel]
TEAM_VALUES     = [t.value for t in TeamLabel]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert software engineering triage assistant.
    Your job is to analyze bug reports and provide structured assessments.
    Always respond with valid JSON only.
    No markdown code fences. No explanation. Just the JSON object.
    Never include text before or after the JSON.
""").strip()


# ---------------------------------------------------------------------------
# Helper — safely extract reward
# ---------------------------------------------------------------------------
def safe_reward(result: Any) -> float:
    r = result.reward
    return float(r) if r is not None else 0.0


# ---------------------------------------------------------------------------
# Helper — build triage prompt
# ---------------------------------------------------------------------------
def build_prompt(obs: BugObservation) -> str:
    comments_text = (
        "\n".join(f"  - {c}" for c in obs.comments)
        if obs.comments else "  (none)"
    )
    labels_text = ", ".join(obs.labels) if obs.labels else "(none)"

    prompt = textwrap.dedent(f"""
        Analyze the following bug report:

        ---
        Issue ID : {obs.issue_id}
        Title    : {obs.title}
        Labels   : {labels_text}

        Description:
        {obs.body}

        Comments:
        {comments_text}
        ---

    """).strip() + "\n\n"

    if obs.task_mode == TaskMode.EASY:
        prompt += textwrap.dedent(f"""
            Task: SEVERITY CLASSIFICATION

            Classify the severity of this bug.
            Valid values: {SEVERITY_VALUES}

            Severity definitions:
              critical -> System down, data loss, security breach, revenue blocked
              high     -> Major feature broken, no workaround
              medium   -> Feature broken but workaround exists
              low      -> Minor / cosmetic issue

            Respond with ONLY this JSON:
            {{
              "severity": "<one of {SEVERITY_VALUES}>"
            }}
        """).strip()

    elif obs.task_mode == TaskMode.MEDIUM:
        prompt += textwrap.dedent(f"""
            Task: SEVERITY + TEAM ROUTING

            1. Classify the severity.
            2. Assign to the correct engineering team.

            Valid severity values: {SEVERITY_VALUES}
            Valid team values: {TEAM_VALUES}
              frontend -> UI, CSS, React, browser, JavaScript
              backend  -> API, database, server-side logic, workers
              security -> Auth, CVEs, data exposure, injection attacks
              devops   -> CI/CD, Docker, Kubernetes, infra, pipelines

            Respond with ONLY this JSON:
            {{
              "severity": "<one of {SEVERITY_VALUES}>",
              "team": "<one of {TEAM_VALUES}>"
            }}
        """).strip()

    else:  # HARD
        prompt += textwrap.dedent(f"""
            Task: FULL TRIAGE — SEVERITY + TEAM + FIX SUGGESTION

            1. Classify the severity.
            2. Assign to the correct engineering team.
            3. Write a detailed technical fix suggestion (minimum 2 sentences).

            Valid severity values: {SEVERITY_VALUES}
            Valid team values: {TEAM_VALUES}
              frontend -> UI, CSS, React, browser, JavaScript
              backend  -> API, database, server-side logic, workers
              security -> Auth, CVEs, data exposure, injection attacks
              devops   -> CI/CD, Docker, Kubernetes, infra, pipelines

            Respond with ONLY this JSON:
            {{
              "severity": "<one of {SEVERITY_VALUES}>",
              "team": "<one of {TEAM_VALUES}>",
              "fix_suggestion": "<your detailed technical fix>"
            }}
        """).strip()

    return prompt


# ---------------------------------------------------------------------------
# Helper — call LLM via OpenAI client
# ---------------------------------------------------------------------------
def call_llm(client: OpenAI, prompt: str) -> dict[str, Any]:
    """
    Call LLM via OpenAI-compatible API.
    Uses API_BASE_URL, MODEL_NAME, HF_TOKEN as per submission requirements.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        content = response.choices[0].message.content or ""
        content = content.strip()

        if DEBUG:
            print(f"      LLM raw: {content[:120]}")

        # Strip markdown fences if model adds them
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
        print(f"      ⚠ LLM returned non-JSON: {e}. Using safe defaults.")
        return {}
    except Exception as e:
        print(f"      ⚠ LLM call failed: {e}. Using safe defaults.")
        return {}


# ---------------------------------------------------------------------------
# Helper — convert LLM response to BugAction sequence
# ---------------------------------------------------------------------------
def build_actions(
    response: dict[str, Any],
    task_mode: TaskMode,
) -> list[BugAction]:
    actions: list[BugAction] = []

    # CLASSIFY
    severity_str = response.get("severity", "medium")
    try:
        severity = SeverityLabel(severity_str)
    except ValueError:
        print(f"      ⚠ Invalid severity '{severity_str}', defaulting to 'medium'")
        severity = SeverityLabel.MEDIUM
    actions.append(BugAction(
        action_type=ActionType.CLASSIFY,
        severity=severity,
    ))

    if task_mode in (TaskMode.MEDIUM, TaskMode.HARD):
        # ASSIGN_TEAM
        team_str = response.get("team", "backend")
        try:
            team = TeamLabel(team_str)
        except ValueError:
            print(f"      ⚠ Invalid team '{team_str}', defaulting to 'backend'")
            team = TeamLabel.BACKEND
        actions.append(BugAction(
            action_type=ActionType.ASSIGN_TEAM,
            team=team,
        ))

    if task_mode == TaskMode.HARD:
        # SUGGEST_FIX
        fix = response.get(
            "fix_suggestion",
            "Investigate the root cause and apply an appropriate fix."
        )
        if len(fix.strip()) < 10:
            fix = "Investigate the root cause and apply an appropriate fix."
        actions.append(BugAction(
            action_type=ActionType.SUGGEST_FIX,
            fix_suggestion=fix,
        ))

    # SUBMIT — always last
    actions.append(BugAction(action_type=ActionType.SUBMIT))

    return actions


# ---------------------------------------------------------------------------
# Async episode runner — matches test.py pattern exactly
# ---------------------------------------------------------------------------
async def run_task(
    client: OpenAI,
    server_url: str,
    task_mode: str,
    seed: int,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run one full episode asynchronously.
    Uses async with MyEnv and await — same pattern as test.py.
    """
    print(f"\n{'='*60}")
    print(f"  TASK: {task_mode.upper()}")
    print(f"  Model: {MODEL_NAME}")
    print(f"{'='*60}")

    start_time = time.time()
    issues_triaged = 0
    step_count = 0

    # ✅ async with + await — same pattern as test.py
    async with MyEnv(base_url=server_url) as env:

        # Reset episode
        result = await env.reset(task_mode=task_mode, seed=seed)
        print(f"  Episode started | First issue: {result.observation.issue_id}")

        while not result.done and step_count < MAX_STEPS:
            obs: BugObservation = result.observation

            # Episode complete
            if obs.issue_id == "DONE":
                break

            if verbose:
                print(f"\n  Issue {obs.issue_id}: {obs.title[:55]}...")
                print(f"  Remaining: {obs.issues_remaining} | Steps: {step_count}")

            # Call LLM to decide actions
            prompt = build_prompt(obs)
            llm_response = call_llm(client, prompt)

            if DEBUG:
                print(f"      LLM parsed: {llm_response}")

            # Build action sequence
            actions = build_actions(llm_response, obs.task_mode)

            # Execute actions
            for action in actions:
                result = await env.step(action)
                step_count += 1

                if verbose and action.action_type == ActionType.SUBMIT:
                    reward = safe_reward(result)
                    print(f"  → Reward: {reward:.3f}")
                    feedback = result.observation.feedback
                    if feedback:
                        print(f"  → {feedback[:80]}...")

                if result.done:
                    break

            issues_triaged += 1

        # Get final state
        state = await env.state()
        elapsed = time.time() - start_time

        print(f"\n  ✅ Episode complete in {elapsed:.1f}s")
        print(f"  Total reward : {state.total_reward:.4f}")
        print(f"  Issues done  : {issues_triaged}/{state.episode_issues}")
        print(f"  Steps taken  : {step_count}")

        avg_score = round(state.total_reward / max(issues_triaged, 1), 4)

        return {
            "task": task_mode,
            "model": MODEL_NAME,
            "api_base_url": API_BASE_URL,
            "total_reward": state.total_reward,
            "avg_score": avg_score,
            "issues_triaged": issues_triaged,
            "episode_issues": state.episode_issues,
            "steps_taken": step_count,
            "per_issue_scores": state.scores_per_issue,
            "elapsed_seconds": round(elapsed, 2),
        }


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------
def print_summary(results: list[dict[str, Any]]) -> None:
    print(f"\n{'='*60}")
    print("  INFERENCE RESULTS — BugTriageEnv")
    print(f"  Model    : {MODEL_NAME}")
    print(f"  API Base : {API_BASE_URL}")
    print(f"{'='*60}")
    print(f"  {'Task':<10} {'Avg Score':<12} {'Total':<10} {'Issues':<10} {'Steps'}")
    print(f"  {'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*8}")
    for r in results:
        issues = f"{r['issues_triaged']}/{r['episode_issues']}"
        print(
            f"  {r['task']:<10} "
            f"{r['avg_score']:<12.4f} "
            f"{r['total_reward']:<10.4f} "
            f"{issues:<10} "
            f"{r['steps_taken']}"
        )
    print(f"{'='*60}")
    all_valid = all(0.0 <= r["avg_score"] <= 1.0 for r in results)
    print(f"\n  Scores in valid range (0.0–1.0): {'✅ Yes' if all_valid else '❌ No'}")
    print()


# ---------------------------------------------------------------------------
# Async main
# ---------------------------------------------------------------------------
async def async_main(args: argparse.Namespace) -> None:
    global DEBUG
    DEBUG = args.debug

    # Validate mandatory env vars
    missing = []
    if not API_BASE_URL:
        missing.append("API_BASE_URL")
    if not MODEL_NAME:
        missing.append("MODEL_NAME")
    if not HF_TOKEN:
        missing.append("HF_TOKEN")

    if missing:
        print(f"\n❌ Missing required environment variables: {missing}")
        print()
        print("  Set them before running:")
        print("  export API_BASE_URL=https://router.huggingface.co/v1")
        print("  export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct")
        print("  export HF_TOKEN=hf_your_token_here")
        sys.exit(1)

    print(f"\n🚀 BugTriageEnv Inference Script")
    print(f"   API Base URL : {API_BASE_URL}")
    print(f"   Model        : {MODEL_NAME}")
    print(f"   Server       : {args.server}")
    print(f"   Seed         : {args.seed}")
    print(f"   Task         : {args.task}")

    # Initialize OpenAI client with mandatory vars
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )

    tasks = (
        ["easy", "medium", "hard"]
        if args.task == "all"
        else [args.task]
    )

    results = []
    total_start = time.time()

    for task in tasks:
        result = await run_task(
            client=client,
            server_url=args.server,
            task_mode=task,
            seed=args.seed,
            verbose=not args.quiet,
        )
        results.append(result)

        # Runtime check — must be under 20 minutes
        elapsed = time.time() - total_start
        if elapsed > 18 * 60:
            print(f"⚠ Approaching 20-minute limit ({elapsed/60:.1f} min elapsed)")

    print_summary(results)

    output_path = "inference_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"💾 Results saved → {output_path}")

    total_elapsed = time.time() - total_start
    print(f"⏱  Total runtime: {total_elapsed/60:.1f} minutes")
    if total_elapsed > 20 * 60:
        print("❌ WARNING: Exceeded 20-minute runtime limit!")
    else:
        print(f"✅ Completed within 20-minute limit")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="BugTriageEnv inference script"
    )
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Task to run (default: all)",
    )
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER,
        help=f"Server URL (default: {DEFAULT_SERVER})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step output",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show LLM raw responses",
    )
    args = parser.parse_args()

    # Run async main
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()