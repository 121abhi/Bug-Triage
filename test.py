"""
Async test script for BugTriageEnv
Run with: python test.py
Server must be running on localhost:8000
"""

import asyncio
from client import MyEnv
from models import (
    ActionType, BugAction, SeverityLabel, TeamLabel
)

SERVER = "http://localhost:8000"
PASSED = []
FAILED = []


def check(name, condition, detail=""):
    if condition:
        print(f"  ✅ {name}")
        PASSED.append(name)
    else:
        print(f"  ❌ {name} {detail}")
        FAILED.append(name)


def safe_reward(result) -> float:
    """Safely extract reward — handles None."""
    r = result.reward
    return float(r) if r is not None else 0.0


# -------------------------------------------------------
async def task_easy():
    print("\n" + "="*50)
    print("  TASK 1 — Easy (Severity Classification)")
    print("="*50)

    async with MyEnv(base_url=SERVER) as env:
        result = await env.reset(task_mode="easy", seed=42)

        check("reset() returns observation",   result.observation is not None)
        check("reset() done=False",            result.done == False)
        check("observation has issue_id",      bool(result.observation.issue_id))
        check("observation has legal_actions", len(result.observation.legal_actions) > 0)
        check("task_mode is easy",             result.observation.task_mode.value == "easy")

        total_reward = 0.0
        steps = 0
        max_steps = 20  # safety guard — prevents infinite loop

        while not result.done and steps < max_steps:
            print(f"    Step {steps+1} | Issue: {result.observation.issue_id} | done={result.done}")

            # Step 1 — CLASSIFY
            result = await env.step(BugAction(
                action_type=ActionType.CLASSIFY,
                severity=SeverityLabel.HIGH
            ))
            print(f"      After CLASSIFY | reward={result.reward} | done={result.done}")

            if result.done:
                break

            # Step 2 — SUBMIT
            result = await env.step(BugAction(action_type=ActionType.SUBMIT))
            print(f"      After SUBMIT   | reward={result.reward} | done={result.done}")

            total_reward += safe_reward(result)
            steps += 1

        if steps >= max_steps:
            print(f"  ⚠ Hit max_steps safety guard ({max_steps})")

        state = await env.state()

        check("Episode completes",             state.is_done)
        check("state() returns total_reward",  state.total_reward is not None)
        check("Reward is numeric",             isinstance(state.total_reward, float))
        check("All issues triaged",            state.current_issue_index >= state.episode_issues)

        print(f"\n  Score: {state.total_reward:.4f} over {steps} issues")


# -------------------------------------------------------
async def task_medium():
    print("\n" + "="*50)
    print("  TASK 2 — Medium (Severity + Team)")
    print("="*50)

    async with MyEnv(base_url=SERVER) as env:
        result = await env.reset(task_mode="medium", seed=42)

        check("reset() works for medium", result.observation is not None)
        check("task_mode is medium",      result.observation.task_mode.value == "medium")

        steps = 0
        max_steps = 30

        while not result.done and steps < max_steps:
            print(f"    Step group {steps+1} | Issue: {result.observation.issue_id}")

            # CLASSIFY
            result = await env.step(BugAction(
                action_type=ActionType.CLASSIFY,
                severity=SeverityLabel.CRITICAL
            ))
            if result.done:
                break

            # ASSIGN_TEAM
            result = await env.step(BugAction(
                action_type=ActionType.ASSIGN_TEAM,
                team=TeamLabel.SECURITY
            ))
            if result.done:
                break

            # SUBMIT
            result = await env.step(BugAction(action_type=ActionType.SUBMIT))
            print(f"      After SUBMIT | reward={result.reward} | done={result.done}")
            steps += 1

        state = await env.state()

        check("Medium episode completes", state.is_done)
        check("Reward in valid range",    -1.0 <= state.total_reward <= 5.0)

        print(f"\n  Score: {state.total_reward:.4f} over {steps} issues")


# -------------------------------------------------------
async def task_hard():
    print("\n" + "="*50)
    print("  TASK 3 — Hard (Full Triage)")
    print("="*50)

    async with MyEnv(base_url=SERVER) as env:
        result = await env.reset(task_mode="hard", seed=42)

        check("reset() works for hard", result.observation is not None)
        check("task_mode is hard",      result.observation.task_mode.value == "hard")

        steps = 0
        max_steps = 40

        while not result.done and steps < max_steps:
            print(f"    Step group {steps+1} | Issue: {result.observation.issue_id}")

            # CLASSIFY
            result = await env.step(BugAction(
                action_type=ActionType.CLASSIFY,
                severity=SeverityLabel.CRITICAL
            ))
            if result.done:
                break

            # ASSIGN_TEAM
            result = await env.step(BugAction(
                action_type=ActionType.ASSIGN_TEAM,
                team=TeamLabel.BACKEND
            ))
            if result.done:
                break

            # SUGGEST_FIX
            result = await env.step(BugAction(
                action_type=ActionType.SUGGEST_FIX,
                fix_suggestion=(
                    "Add idempotency keys to payment API calls. "
                    "Use database-level locking to prevent race conditions "
                    "and ensure atomic updates on order status transitions."
                )
            ))
            if result.done:
                break

            # SUBMIT
            result = await env.step(BugAction(action_type=ActionType.SUBMIT))
            print(f"      After SUBMIT | reward={result.reward} | done={result.done}")
            steps += 1

        state = await env.state()

        check("Hard episode completes",        state.is_done)
        check("per_issue_scores populated",    len(state.scores_per_issue) > 0)
        check("fix_score present in scores",   "fix_score" in state.scores_per_issue[0])

        print(f"\n  Score: {state.total_reward:.4f} over {steps} issues")


# -------------------------------------------------------
async def edge_cases():
    print("\n" + "="*50)
    print("  EDGE CASES")
    print("="*50)

    async with MyEnv(base_url=SERVER) as env:
        result = await env.reset(task_mode="easy", seed=99)

        # SKIP
        result = await env.step(BugAction(action_type=ActionType.SKIP))
        check("SKIP gives negative reward", result.reward is not None and result.reward < 0)
        check("SKIP penalty is -0.2",       result.reward == -0.2)

        # Invalid CLASSIFY (no severity)
        result = await env.step(BugAction(action_type=ActionType.CLASSIFY))
        check("CLASSIFY without severity penalised", result.reward is not None and result.reward < 0)


# -------------------------------------------------------
async def main():
    await task_easy()
    await task_medium()
    await task_hard()
    await edge_cases()

    print("\n" + "="*50)
    print(f"  RESULTS: {len(PASSED)} passed / {len(FAILED)} failed")
    print("="*50)

    if FAILED:
        print(f"\n  ❌ Failed: {FAILED}")
    else:
        print("\n  🎉 All tests passed! Ready for submission.")


if __name__ == "__main__":
    asyncio.run(main())