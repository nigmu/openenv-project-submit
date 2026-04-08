# Customer Service Bot - OpenEnv Environment

A real-world customer service training environment where AI agents learn to handle customer inquiries professionally across three difficulty levels.

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Design Decisions](#design-decisions)
4. [Tasks](#tasks)
5. [Observation Space](#observation-space)
6. [Action Space](#action-space)
7. [Reward Design](#reward-design)
8. [Customer Simulation Logic](#customer-simulation-logic)
9. [Grading Logic](#grading-logic)
10. [Scoring](#scoring)
11. [Project Structure](#project-structure)
12. [Quick Start](#quick-start)
13. [Running and Testing](#running-and-testing)
14. [API Endpoints](#api-endpoints)
15. [LLM Provider Configuration](#llm-provider-configuration)
16. [Docker](#docker)
17. [Environment Variables](#environment-variables)

---

## Overview

This environment simulates realistic customer service interactions for training AI agents. Agents must respond to customer messages with appropriate empathy, accuracy, and professionalism. The environment provides dense reward signals, multi-dimensional grading, and realistic customer behavior simulation.

**Key features:**
- 3 difficulty levels (easy, medium, hard) with 10 unique scenarios
- Dynamic customer simulation with mood, patience, and frustration tracking
- Multi-dimensional grading rubrics (4-5 criteria per task)
- Dense reward signals at every step (not just episode end)
- Deterministic and reproducible scoring with seeds
- LLM-powered or rule-based agent inference (Ollama or OpenAI)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING/                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  inference.py                                         │  │
│  │  ┌─────────────┐    ┌─────────────┐                   │  │
│  │  │ LLM Client  │    │ Rule-Based  │                   │  │
│  │  │ (Ollama/    │    │ Fallback    │                   │  │
│  │  │  OpenAI)    │    │             │                   │  │
│  │  └──────┬──────┘    └──────┬──────┘                   │  │
│  │         └────────┬─────────┘                          │  │
│  │                  ▼                                    │  │
│  │         Action Generator                              │  │
│  └──────────────────┬────────────────────────────────────┘  │
└─────────────────────┼───────────────────────────────────────┘
                      │ HTTP POST /step
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  RUNTIME/ (FastAPI Server)                  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  server/app.py (FastAPI)                            │   │
│  │  POST /reset  POST /step  GET /state  GET /health   │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                   │
│  ┌──────────────────────▼──────────────────────────────┐   │
│  │  server/environment.py (CustomerServiceEnv)         │   │
│  │  ├── reset() → Observation                          │   │
│  │  ├── step(Action) → (Observation, Reward, done, info)│   │
│  │  └── state() → State                                │   │
│  └──────────┬───────────────────────────┬──────────────┘   │
│             │                           │                   │
│  ┌──────────▼──────────┐   ┌───────────▼──────────────┐   │
│  │  src/tasks/         │   │  src/customers/          │   │
│  │  ├── easy_faq.py    │   │  ├── scenarios.py        │   │
│  │  ├── medium_        │   │  └── simulator.py        │   │
│  │  │   complaint.py   │   │     (Customer behavior)  │   │
│  │  └── hard_          │   │                          │   │
│  │      escalation.py  │   │  ┌─────────────────────┐ │   │
│  │                     │   │  │ CustomerState       │ │   │
│  │  ┌──────────────────┤   │  │ - current_mood      │ │   │
│  │  │ src/graders/     │   │  │ - frustration_level │ │   │
│  │  ├── faq_grader.py  │   │  │ - satisfaction_hist │ │   │
│  │  ├── complaint_     │   │  │ - is_escalated      │ │   │
│  │  │   grader.py      │   │  └─────────────────────┘ │   │
│  │  └── escalation_    │   └──────────────────────────┘   │
│  │      grader.py      │                                   │
│  └─────────────────────┘                                   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  src/knowledge_base/                                │   │
│  │  ├── policies.json  (shipping, returns, payments)   │   │
│  │  ├── faqs.json      (8 FAQ entries with keywords)   │   │
│  │  └── procedures.json (5 escalation procedures)      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  src/utils/                                         │   │
│  │  ├── scoring.py   (sentiment, professionalism, etc) │   │
│  │  ├── sentiment.py (sentiment analysis utilities)    │   │
│  │  └── validators.py (input/output validation)        │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Training script** calls `POST /reset` → environment creates a customer scenario
2. Environment returns initial **Observation** (customer message, sentiment, intent, urgency)
3. Training script generates an **Action** (message + action_type) via LLM or rule-based logic
4. Training script calls `POST /step` with the action
5. Environment:
   - Evaluates action against task-specific grader
   - Customer simulator generates a response based on quality and mood
   - Calculates dense reward with multi-component breakdown
   - Returns (Observation, Reward, done, info)
6. Steps repeat until episode ends (max turns, resolution, or customer leaves)

---

## Design Decisions

### Why separate RUNTIME and TRAINING?

- **RUNTIME** is the deployable environment (Docker, HF Spaces). It must be lightweight and self-contained.
- **TRAINING** contains heavy dependencies (PyTorch, transformers) and experiment code. It's not deployed.
- This separation keeps Docker images small and ensures the HF Space loads fast.

### Why multi-dimensional grading instead of binary?

Binary (0/1) scoring provides no learning signal for RL. Our rubrics produce continuous scores in [0.0, 1.0] across 4-5 criteria, giving agents nuanced feedback on what to improve.

### Why dense rewards?

Sparse end-of-episode rewards make RL training extremely slow. Our per-step reward combines:
- Task-specific grading progress (40%)
- Customer sentiment improvement (30%)
- Action type appropriateness (variable)
- Verbosity penalty for overly long responses

This gives the agent a learning signal at every step.

### Why rule-based fallback?

LLMs can fail (network errors, empty responses, rate limits). The rule-based fallback ensures the environment always produces valid actions, making testing and evaluation reliable.

### Customer simulation design

Customers are not static — they have:
- **Mood** (0-10): Changes based on response quality
- **Patience** (0-10): Affects how quickly mood changes
- **Frustration level** (0-1): Increases with poor responses, triggers escalation
- **has_been_addressed flag**: Once a good response is given, the customer stops complaining

This creates realistic dynamics where good agents see improving customer behavior and poor agents see escalating frustration.

---

## Tasks

### Easy: FAQ Response
- **Scenario**: Customers ask common questions about shipping, returns, payments, and policies
- **Objective**: Provide accurate, polite, concise answers
- **Grading**: Correctness (40%), Professionalism (20%), Conciseness (20%), No hallucination (20%)
- **Max turns**: 5
- **Scenarios**: 5 (shipping time, return policy, payment methods, order tracking, warranty)

### Medium: Complaint Resolution
- **Scenario**: Customers have problems (defective products, late deliveries, billing errors)
- **Objective**: Acknowledge issue, show empathy, propose solution, follow up
- **Grading**: Empathy (25%), Problem identification (25%), Solution appropriateness (25%), Professionalism (25%)
- **Max turns**: 7
- **Scenarios**: 3 (defective charger, late birthday gift, double billing charge)

### Hard: Multi-Turn Escalation
- **Scenario**: Complex issues requiring de-escalation and potential supervisor handoff
- **Objective**: Navigate conversation, de-escalate anger, provide resolution or proper handoff
- **Grading**: De-escalation (20%), Information gathering (20%), Resolution path (20%), Satisfaction trajectory (20%), Protocol compliance (20%)
- **Max turns**: 10
- **Scenarios**: 2 (second defective product with manager demand, bulk business order crisis)

---

## Observation Space

```python
class Observation(BaseModel):
    customer_message: str        # The customer's latest message
    sentiment: float             # Current sentiment (-1.0 to 1.0)
    intent: str                  # Detected intent category (e.g., "shipping_inquiry")
    urgency: int                 # Urgency level (1=low to 5=critical)
    conversation_history: list   # Full conversation transcript (alternating customer/agent)
    turn_count: int              # Current turn number (starts at 0)
    task_type: str               # Task difficulty: "easy", "medium", or "hard"
    scenario_description: str    # Brief scenario context for the agent
```

---

## Action Space

```python
class ActionType(str, Enum):
    ANSWER = "answer"           # Direct answer to customer
    ACKNOWLEDGE = "acknowledge" # Acknowledge customer's feelings
    ASK_CLARIFY = "ask_clarify" # Ask clarifying questions
    ESCALATE = "escalate"       # Escalate to supervisor
    CLOSE = "close"             # Close the conversation

class Action(BaseModel):
    message: str                 # Response message to customer
    action_type: ActionType      # One of the 5 action types above
    confidence: float            # Agent confidence (0.0-1.0, default 0.5)
```

---

## Reward Design

Rewards are dense and provided at every step:

### Per-Step Reward Formula

```
reward = (
    0.4 * task_progress_score +      # How well the response addresses the task
    0.3 * sentiment_delta +           # Change in customer mood (normalized)
    action_type_bonus +               # Bonus for appropriate action type
    verbosity_penalty                  # Penalty for overly long responses
)
```

### Action Type Bonuses

| Action Type | Condition | Bonus |
|---|---|---|
| `acknowledge` | Quality > 0.6 | +0.05 |
| `ask_clarify` | Always | +0.03 |
| `escalate` | Frustration > 0.5 | +0.10 |
| `escalate` | Frustration <= 0.5 | -0.05 |
| `close` | Mood >= 6.0 | +0.10 |
| `close` | Mood < 4.0 | -0.10 |

### Penalties

- **Verbosity**: -0.1 if message exceeds 500 characters
- **Empty message**: Episode ends immediately with reward 0.0

### Episode Bonuses

- +0.2 added to final step reward if customer issue is resolved

---

## Customer Simulation Logic

### Customer State

Each customer has these tracked attributes:

| Attribute | Range | Description |
|---|---|---|
| `current_mood` | 0.0 - 10.0 | 0 = furious, 10 = delighted |
| `frustration_level` | 0.0 - 1.0 | Accumulates with poor responses |
| `has_been_addressed` | bool | Set to True when quality >= 0.6 |
| `is_escalated` | bool | True when frustration > 0.7 and RNG triggers |
| `is_resolved` | bool | Set when issue is fully resolved |
| `satisfaction_history` | list[float] | Mood after each turn |

### Mood Change Formula

```python
base_delta = (quality - 0.5) * 2.0

# Action type modifiers
if action == "acknowledge" and quality > 0.6: base_delta += 0.3
if action == "escalate" and frustration > 0.5: base_delta += 0.5
if action == "close" and mood < 4.0: base_delta -= 1.5

# Length modifiers
if len(message) > 500: base_delta -= 0.2
if len(message) < 10: base_delta -= 0.3

# Scale by customer patience
final_delta = base_delta * (patience / 10.0)
```

### Response Selection Logic

The customer selects responses based on quality and mood:

| Quality | Mood >= 7 | Mood 4-7 | Mood < 4 |
|---|---|---|---|
| >= 0.7 | "Thank you, that's helpful!" | "Thank you, that's helpful!" | "Can you tell me more?" |
| 0.4-0.7 | "Okay." / "I see." | "Okay." / follow-up | "That's not what I asked." |
| < 0.4 | "That's not what I asked." | "That's not what I asked." | "This is frustrating." / "This is unacceptable!" |

### Escalation Trigger

When `frustration_level > 0.7`, there's a 40% chance per turn that `is_escalated` becomes True, shifting customer responses to angry templates.

---

## Grading Logic

### Easy (FAQ) Grader

```
correctness = keyword_coverage(key_facts) * 0.6 + faq_keyword_match * 0.4
professionalism = count(professional_phrases) / count(all_phrases)
conciseness = 1.0 if 30 <= len <= 250 else scaled
hallucination = verified_claims / total_claims
final = (correctness + professionalism + conciseness + hallucination) / 4
```

### Medium (Complaint) Grader

```
empathy = min(1.0, count(empathy_indicators) / 3.0)
problem_id = keyword_coverage(customer_key_facts)
solution = matched_procedure_steps / total_procedure_steps
professionalism = count(professional_phrases) / count(all_phrases)
final = (empathy + problem_id + solution + professionalism) / 4
```

### Hard (Escalation) Grader

```
de_escalation = empathy_phrases_score + action_type_bonus + timing_bonus + de_escalation_attempts
info_gathering = key_facts_mentioned * 0.5 + question_score + context_awareness
resolution_path = escalation_appropriateness + resolution_actions_mentioned
satisfaction_trajectory = recent_positive_vs_negative_signals
protocol_compliance = procedure_steps_followed + escalation_timing
final = (de_escalation + info_gathering + resolution_path + satisfaction_trajectory + protocol_compliance) / 5
```

---

## Scoring

All graders produce scores in the range [0.0, 1.0] with multi-dimensional breakdowns:

| Task | Criteria | Weights |
|---|---|---|
| **Easy** | Correctness, Professionalism, Conciseness, Hallucination | 25% each |
| **Medium** | Empathy, Problem ID, Solution, Professionalism | 25% each |
| **Hard** | De-escalation, Info Gathering, Resolution Path, Satisfaction Trajectory, Protocol Compliance | 20% each |

Scoring is deterministic and reproducible with the same seed.

---

## Project Structure

### Full Repository Layout

```
openenv-project/
├── API_CREDENTIALS.py              # API keys (gitignored)
├── ALL_API                         # Alternative API key file (gitignored)
├── .gitignore
├── inference.py                    # Root-level baseline script (hackathon requirement)
│
├── RUNTIME/
│   └── customer-service-bot/
│       ├── server/
│       │   ├── __init__.py
│       │   ├── app.py              # FastAPI server with logging middleware
│       │   └── environment.py      # CustomerServiceEnv (reset/step/state)
│       │
│       ├── src/
│       │   ├── tasks/
│       │   │   ├── __init__.py
│       │   │   ├── base.py         # BaseTask, TaskResult
│       │   │   ├── easy_faq.py     # FAQTask + FAQGrader
│       │   │   ├── medium_complaint.py  # ComplaintTask + ComplaintGrader
│       │   │   └── hard_escalation.py   # EscalationTask + EscalationGrader
│       │   │
│       │   ├── graders/
│       │   │   ├── __init__.py
│       │   │   └── base.py         # BaseGrader interface
│       │   │
│       │   ├── customers/
│       │   │   ├── __init__.py
│       │   │   ├── scenarios.py    # 10 customer scenarios across 3 difficulties
│       │   │   └── simulator.py    # CustomerSimulator with mood/frustration logic
│       │   │
│       │   ├── knowledge_base/
│       │   │   ├── __init__.py
│       │   │   ├── policies.json   # Company policies
│       │   │   ├── faqs.json       # 8 FAQ entries
│       │   │   └── procedures.json # 5 escalation procedures
│       │   │
│       │   └── utils/
│       │       ├── __init__.py
│       │       ├── scoring.py      # Sentiment, professionalism, empathy, conciseness, hallucination
│       │       ├── sentiment.py    # Sentiment analysis utilities
│       │       └── validators.py   # Action validation
│       │
│       ├── tests/
│       │   ├── __init__.py
│       │   └── test_environment.py # 8 comprehensive tests
│       │
│       ├── models.py               # Pydantic: Observation, Action, Reward, State
│       ├── client.py               # OpenAI client wrapper
│       ├── openenv.yaml            # Environment metadata
│       ├── requirements.txt        # Runtime dependencies
│       ├── start_server.py         # Server launcher
│       ├── Dockerfile              # Container config
│       └── README.md               # This file
│


---

## Quick Start

### Prerequisites

- Python 3.10+
- pip
- A virtual environment (`.venv`) — **must be created for your OS** (WSL venv won't work on Windows and vice versa)

### First-Time Setup

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
# WSL/Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Install dependencies
cd RUNTIME/customer-service-bot
pip install -r requirements.txt
```

---

## Running and Testing

### Step 1: Start the Server

Open a terminal and run:

**WSL / Linux / macOS:**
```bash
cd RUNTIME/customer-service-bot
source .venv/bin/activate
uvicorn server.app:app --host 127.0.0.1 --port 8000
```

**Windows PowerShell:**
```powershell
cd RUNTIME\customer-service-bot
.\.venv\Scripts\Activate.ps1
uvicorn server.app:app --host 127.0.0.1 --port 8000
```

If PowerShell execution policy blocks activation:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Expected output:
```
INFO:     Started server process [xxxx]
INFO:     Waiting for application startup.
INFO:     Environment initialized
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Step 2: Run the Test Suite

Open a **second terminal** (keep the server running in the first):

```bash
cd RUNTIME/customer-service-bot
source .venv/bin/activate   # or .\.venv\Scripts\Activate.ps1 on Windows
python tests/test_environment.py
```

Expected: **8 passed, 0 failed**

### Step 3: Run Baseline Inference

```bash
cd TRAINING
source .venv/bin/activate   # or .\.venv\Scripts\Activate.ps1 on Windows
python inference.py
```

This runs all 3 tasks and prints full conversations, rewards, and scores.

---

## API Endpoints

### `GET /health`

Returns `{"status": "ok"}`. Used for health checks.

### `POST /reset`

Resets the environment and starts a new episode.

**Request body:**
```json
{
  "task_type": "easy",
  "scenario_index": 0,
  "seed": 42
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `task_type` | string | random | "easy", "medium", or "hard" |
| `scenario_index` | int | 0 | Which scenario within the difficulty |
| `seed` | int | null | Random seed for reproducibility |

**Response:** `Observation` object

### `POST /step`

Takes an action and returns the next observation, reward, and done flag.

**Request body:**
```json
{
  "message": "We offer Standard (5-7 business days, $5.99)...",
  "action_type": "answer",
  "confidence": 0.8
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `message` | string | required | Agent's response to customer |
| `action_type` | string | required | "answer", "acknowledge", "ask_clarify", "escalate", "close" |
| `confidence` | float | 0.5 | Agent confidence (0.0-1.0) |

**Response:** `StepResponse` with observation, reward, done, and info.

### `GET /state`

Returns the current environment state.

**Response:** `State` object with customer_mood, satisfaction_trajectory, resolution_status, turn_count, task_type, conversation_history, episode_done.

### Server Logging

Every request is logged with detailed information:
```
19:30:15 | INFO | POST /reset | status=200 | 12ms
19:30:15 | INFO |   -> reset: task=easy, scenario=0, seed=42
19:30:15 | INFO |   <- reset: task=easy, customer="Hi! I'm thinking..."
19:30:15 | INFO | POST /step | status=200 | 8ms
19:30:15 | INFO |   -> step: action=answer, message="We offer Standard..."
19:30:15 | INFO |   <- step: reward=0.332, task_score=0.795, done=False, mood=7.5
```

---

## LLM Provider Configuration

### API Credentials

Store your API keys in `API_CREDENTIALS.py` at the project root:

```python
OLLAMA_API_KEY = "your-ollama-key"
OPENAI_API_KEY = "sk-your-openai-key"
```

This file is **gitignored** — never commit real keys.

### Switching Providers

The inference script reads `API_CREDENTIALS.py` automatically. Switch providers via environment variable:

**Use Ollama (default):**
```powershell
$env:LLM_PROVIDER="ollama"
$env:LLM_MODEL="minimax-m2.7:cloud"
python inference.py
```

**Use OpenAI:**
```powershell
$env:LLM_PROVIDER="openai"
$env:LLM_MODEL="gpt-3.5-turbo"
python inference.py
```

**WSL/Linux:**
```bash
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-3.5-turbo
python inference.py
```

### Startup Banner

```
============================================================
Customer Service Bot - Baseline Inference
  LLM Provider : OPENAI
  LLM Model    : gpt-3.5-turbo
  LLM API URL  : https://api.openai.com/v1
  Env Server   : http://localhost:8000
  LLM Active   : YES
============================================================
```

### LLM Fallback Behavior

If the LLM call fails (network error, empty response, rate limit):
1. Retries once with lower temperature (0.3 instead of 0.7)
2. Falls back to rule-based responses if retry also fails
3. Prints a warning message so you know fallback was used

---

## Docker

```bash
# From RUNTIME/customer-service-bot/
docker build -t customer-service-bot .
docker run -p 8000:8000 customer-service-bot
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `"ollama"` | LLM backend: "ollama" or "openai" |
| `LLM_MODEL` | `"minimax-m2.7:cloud"` (ollama) / `"gpt-3.5-turbo"` (openai) | Model name |
| `LLM_API_BASE` | `"http://localhost:11434/v1"` (ollama) / `"https://api.openai.com/v1"` (openai) | LLM API endpoint |
| `ENV_BASE_URL` | `"http://localhost:8000"` | Environment server URL |
| `OLLAMA_API_KEY` | `"ollama"` | Ollama API key (from `API_CREDENTIALS.py`) |
| `OPENAI_API_KEY` | `""` | OpenAI API key (from `API_CREDENTIALS.py`) |
| `HF_TOKEN` | `""` | Hugging Face token (alternative to OPENAI_API_KEY) |

---

## Test Suite

Run with `python tests/test_environment.py`. Covers:

| Test | What It Validates |
|---|---|
| Reset all task types | reset() works for easy/medium/hard |
| Easy task full episode | Complete episode with valid rewards |
| Medium task full episode | Complete episode with valid rewards |
| Hard task full episode | Complete episode with valid rewards |
| State management | state() returns correct data |
| Reward range | All rewards in [0.0, 1.0] |
| Deterministic scoring | Same seed produces identical results |
| Episode boundaries | Episodes end at correct turn counts |
