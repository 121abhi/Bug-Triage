# 🐛 BugTriageEnv

> A real-world OpenEnv environment where an AI agent acts as a software engineering triage assistant.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](http://meta-pytorch.org/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.11+-green)](https://www.python.org)
[![Docker](https://img.shields.io/badge/Docker-required-blue)](https://www.docker.com)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange)](LICENSE)

---

## 📖 Overview

**BugTriageEnv** simulates a real-world task that software engineers perform every day — reading incoming bug reports and deciding how to handle them. The agent must:

- **Classify** the severity of each bug (`critical / high / medium / low`)
- **Route** the issue to the correct engineering team (`frontend / backend / security / devops`)
- **Suggest** a concrete technical fix for the root cause

The environment is fully compliant with the [OpenEnv spec](http://meta-pytorch.org/OpenEnv) and exposes the standard `reset()` / `step()` / `state()` API. It runs as a containerized FastAPI server behind a WebSocket interface.

### Why Bug Triage?

Bug triage is a high-value, knowledge-intensive task that every engineering organization struggles with. It requires:
- Domain knowledge (what makes a bug critical vs. low?)
- Organizational knowledge (which team owns what?)
- Technical reasoning (what is the likely root cause and fix?)

This makes it an ideal benchmark for evaluating LLM agents on real-world reasoning.

---

## 🎯 Tasks

BugTriageEnv includes **3 tasks of increasing difficulty**, each with a programmatic grader that scores agent performance from `0.0` to `1.0`.

### Task 1 — Severity Classification (Easy)

| Property | Value |
|---|---|
| Difficulty | Easy |
| Issues per episode | 5 |
| Max steps | 15 |
| Agent must | Classify each bug as `critical / high / medium / low` |
| Reward | Partial credit via severity neighbour map |
| Baseline score (random) | ~0.42 |

**Agent flow:**
```
CLASSIFY (severity) → SUBMIT
```

**Scoring:**
```
Exact match         → 1.0
One level off       → 0.5   (e.g. HIGH when truth is CRITICAL)
Two levels off      → 0.2
Completely wrong    → 0.0   (e.g. LOW when truth is CRITICAL)
```

---

### Task 2 — Severity + Team Routing (Medium)

| Property | Value |
|---|---|
| Difficulty | Medium |
| Issues per episode | 5 |
| Max steps | 20 |
| Agent must | Classify severity AND assign to correct team |
| Reward | severity(0.5) + team(0.5) |
| Baseline score (random) | ~0.28 |

**Agent flow:**
```
CLASSIFY (severity) → ASSIGN_TEAM → SUBMIT
```

**Scoring:**
```
severity_score × 0.5   (partial credit)
team_score     × 0.5   (binary: correct=1.0, wrong=0.0)
```

---

### Task 3 — Full Triage Workflow (Hard)

| Property | Value |
|---|---|
| Difficulty | Hard |
| Issues per episode | 5 |
| Max steps | 30 |
| Agent must | Classify + route + write a concrete fix suggestion |
| Reward | severity(0.3) + team(0.3) + fix(0.4) |
| Baseline score (random) | ~0.18 |

**Agent flow:**
```
CLASSIFY (severity) → ASSIGN_TEAM → SUGGEST_FIX → SUBMIT
```

**Scoring:**
```
severity_score × 0.3   (partial credit)
team_score     × 0.3   (binary)
fix_score      × 0.4   (keyword match ratio against ground truth)
```

Fix score methodology: if the agent's suggestion contains ≥40% of the ground truth fix keywords, it receives full fix score. Partial credit is given proportionally below this threshold.

---

## 🔧 Action Space

The agent can take **5 action types**. Legal actions per step are always listed in `observation.legal_actions`.

| Action | Required Field | Description |
|---|---|---|
| `classify` | `severity` | Assign a severity label to the current issue |
| `assign_team` | `team` | Route the issue to the correct team |
| `suggest_fix` | `fix_suggestion` | Write a free-text fix suggestion (min 10 chars) |
| `submit` | — | Finalize issue, trigger grader, receive reward |
| `skip` | — | Skip issue with -0.2 penalty |

**Severity values:** `critical`, `high`, `medium`, `low`

**Team values:** `frontend`, `backend`, `security`, `devops`

---

## 👁️ Observation Space

After every `reset()` or `step()`, the agent receives a structured `BugObservation`:

| Field | Type | Description |
|---|---|---|
| `issue_id` | `str` | Unique issue identifier (e.g. `BUG-001`) |
| `title` | `str` | Short one-line title of the bug |
| `body` | `str` | Full bug description |
| `labels` | `list[str]` | Pre-existing labels on the issue |
| `comments` | `list[str]` | Developer/user comments on the issue |
| `task_mode` | `enum` | Current task: `easy` / `medium` / `hard` |
| `step_count` | `int` | Steps taken so far this episode |
| `issues_remaining` | `int` | Issues left to triage |
| `last_reward` | `float` | Reward from the last action |
| `feedback` | `str` | Human-readable grader feedback |
| `legal_actions` | `list[str]` | Valid actions the agent can take right now |

---

## 🏆 Reward Function

BugTriageEnv uses a **partial progress reward** — the agent receives reward after every issue, not just at episode end. This provides dense learning signal throughout training.

```
EASY   reward = severity_score                              (max 1.0)
MEDIUM reward = severity_score × 0.5 + team_score × 0.5   (max 1.0)
HARD   reward = severity_score × 0.3 +
                team_score     × 0.3 +
                fix_score      × 0.4                       (max 1.0)

Penalties:
  skip_penalty  = -0.2  (when agent calls SKIP)
  step_penalty  = -0.05 (per extra step beyond the minimum needed)
```

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.11+
- Docker Desktop (running)
- `uv` package manager
- Git

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/bug-triage-env
cd bug-triage-env
```

### Step 2 — Install OpenEnv

```bash
pip install openenv-core
```

### Step 3 — Install dependencies

```bash
cd my_env
uv sync
```

### Step 4 — Generate the dataset

```bash
python generate_data.py
```

This creates `server/data/bug_issues.json` with 15 synthetic bug issues.

### Step 5 — Build the Docker image

```bash
openenv build
```

### Step 6 — Start the server

```bash
# Option A: Docker (recommended)
docker run -p 8000:8000 openenv-my-env:latest

# Option B: Direct with uv
uv run --project . server --port 8000
```

### Step 7 — Validate

```bash
openenv validate
```

Expected output:
```
[PASS] my_env: Ready for multi-mode deployment
```

---

## 🤖 Running the Baseline

The baseline script runs a **Gemini-powered LLM agent** against all 3 tasks.

### Step 1 — Get a free Gemini API key

1. Go to [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Click **Create API Key**
3. Copy the key

### Step 2 — Set up environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your-key-here
```

### Step 3 — Install dependencies

```bash
pip install openai python-dotenv
```

### Step 4 — Run the baseline

```bash
# Run all 3 tasks
python baseline.py

# Run a specific task
python baseline.py --task easy
python baseline.py --task medium
python baseline.py --task hard

# Use a different Gemini model
python baseline.py --model gemini-2.5-flash

# Reproducible run with fixed seed
python baseline.py --seed 42
```

### Baseline Results (Gemini 2.0 Flash, seed=42)

| Task | Avg Score | Total Reward | Issues | Steps |
|---|---|---|---|---|
| easy | ~0.82 | ~4.10 | 5/5 | 10 |
| medium | ~0.61 | ~3.05 | 5/5 | 15 |
| hard | ~0.44 | ~2.20 | 5/5 | 20 |

*Scores may vary slightly depending on Gemini model version.*

---

## 💻 Usage — Python API

```python
from my_env.client import MyEnv
from my_env.models import BugAction, ActionType, SeverityLabel, TeamLabel

# Connect to the running server
with MyEnv(base_url="http://localhost:8000") as env:

    # Reset for Task 1 (Easy)
    result = env.reset({"task_mode": "easy", "seed": 42})

    while not result.done:
        obs = result.observation
        print(f"Issue: {obs.title}")
        print(f"Legal actions: {obs.legal_actions}")

        # Task 1: classify then submit
        result = env.step(BugAction(
            action_type=ActionType.CLASSIFY,
            severity=SeverityLabel.HIGH
        ))
        result = env.step(BugAction(
            action_type=ActionType.SUBMIT
        ))
        print(f"Reward: {result.reward}")
        print(f"Feedback: {result.observation.feedback}")

    # Get final state
    state = env.state()
    print(f"Total reward: {state.total_reward}")
    print(f"Per-issue scores: {state.scores_per_issue}")
```

### Task 2 (Medium) example

```python
result = env.reset({"task_mode": "medium", "seed": 42})
while not result.done:
    env.step(BugAction(action_type=ActionType.CLASSIFY,
                       severity=SeverityLabel.CRITICAL))
    env.step(BugAction(action_type=ActionType.ASSIGN_TEAM,
                       team=TeamLabel.SECURITY))
    result = env.step(BugAction(action_type=ActionType.SUBMIT))
```

### Task 3 (Hard) example

```python
result = env.reset({"task_mode": "hard", "seed": 42})
while not result.done:
    env.step(BugAction(action_type=ActionType.CLASSIFY,
                       severity=SeverityLabel.CRITICAL))
    env.step(BugAction(action_type=ActionType.ASSIGN_TEAM,
                       team=TeamLabel.BACKEND))
    env.step(BugAction(action_type=ActionType.SUGGEST_FIX,
                       fix_suggestion="Add idempotency keys to payment API calls "
                                      "and use optimistic locking to prevent race conditions."))
    result = env.step(BugAction(action_type=ActionType.SUBMIT))
```

---

## 📁 Project Structure

```
my_env/
├── models.py                    # Pydantic models (Observation, Action, State, Reward)
├── client.py                    # Typed WebSocket client
├── openenv.yaml                 # OpenEnv manifest
├── baseline.py                  # Gemini baseline inference script
├── generate_data.py             # Dataset generation script
├── README.md                    # This file
├── pyproject.toml               # Package config
├── .env                         # API keys (never commit this)
└── server/
    ├── app.py                   # FastAPI server
    ├── my_env_environment.py    # Core environment logic + graders
    ├── requirements.txt         # Server dependencies
    ├── Dockerfile               # Container definition
    └── data/
        └── bug_issues.json      # 15 synthetic bug issues
```

---

## 🗂️ Dataset

The environment uses **15 synthetic GitHub-style bug issues** across 3 difficulty tiers:

| Issues | Severity Distribution | Teams Covered |
|---|---|---|
| BUG-001 to BUG-005 (Easy) | low × 2, medium × 1, high × 1, critical × 1 | frontend, backend |
| BUG-006 to BUG-010 (Medium) | critical × 2, high × 3 | security, devops, frontend, backend |
| BUG-011 to BUG-015 (Hard) | critical × 4, high × 1 | backend × 2, security × 2, devops × 1 |

Regenerate the dataset anytime:
```bash
python generate_data.py
```

---

## 🐳 Docker

```bash
# Build
openenv build

# Run
docker run -p 8000:8000 openenv-my-env:latest

# Check logs
docker logs <container-id>
```

---

## ☁️ Deploy to Hugging Face

```bash
# Login to Hugging Face
huggingface-cli login

# Push environment
openenv push

# Push to a specific repo
openenv push --repo-id your-username/bug-triage-env

# Push as private
openenv push --private
```

Once deployed, others can use your environment with:

```python
env = MyEnv.from_hub("your-username/bug-triage-env")
```

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE) for details.