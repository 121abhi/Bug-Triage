"""
Inference Script — BugTriageEnv
===================================
MANDATORY requirements met:
    - Named inference.py, placed in root directory
    - Uses API_BASE_URL, MODEL_NAME, HF_TOKEN env vars
    - Uses OpenAI Client for all LLM calls
    - Emits [START], [STEP], [END] structured stdout logs
    - Must complete in under 20 minutes
    - Runs on vcpu=2, memory=8gb

Environment variables:
    API_BASE_URL        The API endpoint for the LLM
    MODEL_NAME          The model identifier
    HF_TOKEN            Your Hugging Face / API key
    LOCAL_IMAGE_NAME    Local Docker image name (optional)

STDOUT FORMAT (strictly followed):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

Example:
    [START] task=easy env=bug-triage-env model=meta-llama/Llama-3.1-8B-Instruct
    [STEP] step=1 action=classify:critical reward=0.00 done=false error=null
    [STEP] step=2 action=submit reward=1.00 done=false error=null
    [END] success=true steps=10 rewards=0.00,1.00,...
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
import time
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# MANDATORY environment variables
# ---------------------------------------------------------------------------
API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # optional Docker image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BENCHMARK              = "bug-triage-env"
MAX_STEPS              = 50
TEMPERATURE            = 0.1
MAX_TOKENS             = 512
SUCCESS_SCORE_THRESHOLD = 0.3   # avg reward >= 0.3 = success

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI
except ImportError:
    print("❌ openai required. Run: pip install openai", flush=True)
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
    print(f"❌ Import failed: {e}", flush=True)
    sys.exit(1)

SEVERITY_VALUES = [s.value for s in SeverityLabel]
TEAM_VALUES     = [t.value for t in TeamLabel]

# ---------------------------------------------------------------------------
# MANDATORY structured log functions — exact format required by judges
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    """Emit [START] line — exactly one per episode."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    """
    Emit [STEP] line — exactly one per env.step() call.
    reward: formatted to 2 decimal places
    done: lowercase boolean string
    error: raw error string or 'null'
    """
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: list[float],
) -> None:
    """
    Emit [END] line — exactly one per episode, always emitted even on exception.
    rewards: comma-separated, 2 decimal places each
    """
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert software engineering triage assistant.
    Analyze bug reports and provide structured assessments.
    Always respond with valid JSON only.
    No markdown code fences. No explanation. Just the JSON object.
""").strip()


# ---------------------------------------------------------------------------
# Helper — safely extract reward
# ---------------------------------------------------------------------------
def safe_reward(result: Any) -> float:
    r = result.reward
    return float(r) if r is not None else 0.0


# ---------------------------------------------------------------------------
# Helper — build prompt from observation
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

            Classify the severity. Valid values: {SEVERITY_VALUES}
              critical -> System down, data loss, security breach
              high     -> Major feature broken, no workaround
              medium   -> Feature broken but workaround exists
              low      -> Minor / cosmetic issue

            Respond with ONLY this JSON:
            {{ "severity": "<one of {SEVERITY_VALUES}>" }}
        """).strip()

    elif obs.task_mode == TaskMode.MEDIUM:
        prompt += textwrap.dedent(f"""
            Task: SEVERITY + TEAM ROUTING

            1. Classify severity. Valid: {SEVERITY_VALUES}
            2. Assign team. Valid: {TEAM_VALUES}
              frontend -> UI, CSS, React, browser
              backend  -> API, database, server-side
              security -> Auth, CVEs, data exposure
              devops   -> CI/CD, Docker, Kubernetes

            Respond with ONLY this JSON:
            {{ "severity": "<value>", "team": "<value>" }}
        """).strip()

    else:  # HARD
        prompt += textwrap.dedent(f"""
            Task: FULL TRIAGE — SEVERITY + TEAM + FIX

            1. Classify severity. Valid: {SEVERITY_VALUES}
            2. Assign team. Valid: {TEAM_VALUES}
            3. Write a detailed technical fix (min 2 sentences).

            Respond with ONLY this JSON:
            {{
              "severity": "<value>",
              "team": "<value>",
              "fix_suggestion": "<detailed fix>"
            }}
        """).strip()

    return prompt


# ---------------------------------------------------------------------------
# Helper — call LLM via OpenAI client
# ---------------------------------------------------------------------------
def call_llm(client: OpenAI, prompt: str) -> dict[str, Any]:
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

    except json.JSONDecodeError:
        return {}
    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", flush=True)
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
# Helper — action to string for [STEP] log
# ---------------------------------------------------------------------------
def action_to_str(action: BugAction) -> str:
    """Convert BugAction to compact string for [STEP] log."""
    if action.action_type == ActionType.CLASSIFY:
        return f"classify:{action.severity.value if action.severity else 'none'}"
    elif action.action_type == ActionType.ASSIGN_TEAM:
        return f"assign_team:{action.team.value if action.team else 'none'}"
    elif action.action_type == ActionType.SUGGEST_FIX:
        # Truncate fix for log readability
        fix = (action.fix_suggestion or "")[:50].replace(" ", "_")
        return f"suggest_fix:{fix}"
    elif action.action_type == ActionType.SUBMIT:
        return "submit"
    elif action.action_type == ActionType.SKIP:
        return "skip"
    return action.action_type.value


# ---------------------------------------------------------------------------
# Async episode runner — one task
# ---------------------------------------------------------------------------
async def run_task(
    client: OpenAI,
    server_url: str,
    task_mode: str,
    seed: int,
) -> dict[str, Any]:
    """
    Run one full episode for a given task.
    Emits [START], [STEP]×n, [END] to stdout.
    """
    rewards: list[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    # Log [START]
    log_start(task=task_mode, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with MyEnv(base_url=server_url) as env:
            result = await env.reset(task_mode=task_mode, seed=seed)

            step_counter = 0

            while not result.done and step_counter < MAX_STEPS:
                obs: BugObservation = result.observation

                if obs.issue_id == "DONE":
                    break

                # Get LLM decision
                prompt   = build_prompt(obs)
                llm_resp = call_llm(client, prompt)
                actions  = build_actions(llm_resp, obs.task_mode)

                # Execute each action and emit [STEP] for each
                for action in actions:
                    result = await env.step(action)
                    step_counter += 1

                    reward    = safe_reward(result)
                    done      = result.done
                    action_str = action_to_str(action)

                    # Get any error from feedback
                    feedback  = result.observation.feedback or ""
                    error_str = None
                    if "Invalid" in feedback or "requires" in feedback.lower():
                        error_str = feedback[:80].replace(" ", "_")

                    # Log [STEP] — one per env.step() call
                    log_step(
                        step=step_counter,
                        action=action_str,
                        reward=reward,
                        done=done,
                        error=error_str,
                    )

                    rewards.append(reward)

                    if done:
                        break

            steps_taken = step_counter

            # Get final state for score calculation
            state = await env.state()
            total_reward = state.total_reward
            num_issues   = max(state.episode_issues, 1)

            # Normalize score to 0.0–1.0
            score   = min(max(total_reward / num_issues, 0.0), 1.0)
            success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
        score   = 0.0
        success = False

    finally:
        # Log [END] — always emitted, even on exception
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )

    return {
        "task":         task_mode,
        "model":        MODEL_NAME,
        "score":        score,
        "total_reward": sum(rewards),
        "avg_reward":   round(sum(rewards) / max(len([r for r in rewards if r != 0]), 1), 4),
        "steps":        steps_taken,
        "success":      success,
        "rewards":      rewards,
    }


# ---------------------------------------------------------------------------
# Async main
# ---------------------------------------------------------------------------
async def async_main(
    tasks: list[str],
    server_url: str,
    seed: int,
) -> None:
    # Validate mandatory env vars
    missing = []
    if not API_BASE_URL:
        missing.append("API_BASE_URL")
    if not MODEL_NAME:
        missing.append("MODEL_NAME")
    if not HF_TOKEN:
        missing.append("HF_TOKEN")

    if missing:
        print(f"[DEBUG] Missing env vars: {missing}", flush=True)
        sys.exit(1)

    # Initialize OpenAI client with mandatory vars
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )

    total_start = time.time()
    all_results = []

    for task in tasks:
        result = await run_task(
            client=client,
            server_url=server_url,
            task_mode=task,
            seed=seed,
        )
        all_results.append(result)

        # Runtime guard
        elapsed = time.time() - total_start
        if elapsed > 18 * 60:
            print(
                f"[DEBUG] Approaching 20-minute limit "
                f"({elapsed/60:.1f} min)",
                flush=True,
            )

    # Save results
    with open("inference_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    total_elapsed = time.time() - total_start
    print(
        f"[DEBUG] Total runtime: {total_elapsed/60:.1f} min",
        flush=True,
    )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

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
        default="http://localhost:8000",
        help="Server URL",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    tasks = (
        ["easy", "medium", "hard"]
        if args.task == "all"
        else [args.task]
    )

    asyncio.run(async_main(
        tasks=tasks,
        server_url=args.server,
        seed=args.seed,
    ))


if __name__ == "__main__":
    main()
