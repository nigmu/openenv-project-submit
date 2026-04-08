"""
Baseline inference for Customer Service Bot (OpenEnv HTTP runtime).

MANDATORY (hackathon)
- Define in environment:
    API_BASE_URL   — LLM API endpoint (OpenAI-compatible).
    MODEL_NAME     — Model id for inference.
    HF_TOKEN       — API key (HF / OpenAI-compatible).
    ENV_BASE_URL   — Customer-service environment server (default http://localhost:8000).
  Optional:
    BENCHMARK      — Name shown in [START] env=... (default: customer-service-bot).

STDOUT: only these line types, in order per episode:
    [START] task=<name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>  (score 3 decimals per sample)

All LLM calls use the OpenAI client with API_BASE_URL, MODEL_NAME, HF_TOKEN.
"""

from __future__ import annotations

import os
import sys
from typing import Any, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Optional: load API_CREDENTIALS.py at project root (gitignored)
# ---------------------------------------------------------------------------
_api_creds_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "API_CREDENTIALS.py")
# Only fill missing keys so explicit env (e.g. HF_TOKEN=) wins over the file.
if os.path.exists(_api_creds_path):
    with open(_api_creds_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value

# ---------------------------------------------------------------------------
# Configuration (all paths relative to repo root; run: python inference.py)
# Hackathon names: API_BASE_URL, MODEL_NAME, HF_TOKEN (OpenAI client).
# Local Ollama demo: set OLLAMA_API_KEY or HF_TOKEN (often "ollama"); MODEL_NAME e.g. llama3.2
# ---------------------------------------------------------------------------

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:8000")
# OpenAI-compatible Ollama endpoint (see https://github.com/ollama/ollama/blob/main/docs/openai.md)
_DEFAULT_LLM_BASE = "http://127.0.0.1:11434/v1"
API_BASE_URL = os.getenv("API_BASE_URL", os.getenv("LLM_API_BASE", _DEFAULT_LLM_BASE))
MODEL_NAME = os.getenv("MODEL_NAME", os.getenv("OLLAMA_MODEL", "llama3.2"))
# HF_TOKEN is required for judging; Ollama accepts a placeholder — map OLLAMA_API_KEY for demos
HF_TOKEN = (
    os.getenv("HF_TOKEN")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("OLLAMA_API_KEY")
    or "ollama"
)
BENCHMARK = os.getenv("BENCHMARK", "customer-service-bot")

SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.1"))
RESET_SEED = int(os.getenv("RESET_SEED", "42"))

# openenv.yaml task names
TASK_NAMES = {
    "easy": "easy_faq",
    "medium": "medium_complaint",
    "hard": "hard_escalation",
}

USE_LLM = bool(HF_TOKEN and "your-key" not in HF_TOKEN.lower() and HF_TOKEN != "dummy")
OPENAI_CLIENT: Optional[OpenAI] = None
if USE_LLM:
    # OpenAI SDK; base_url points at Ollama or any OpenAI-compatible server
    OPENAI_CLIENT = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    if error:
        err = error.replace("\n", " ").replace("\r", " ")[:300]
    else:
        err = "null"
    # Escape so one line: no raw newlines in action
    action_one = action.replace("\n", " ").replace("\r", " ")
    if len(action_one) > 800:
        action_one = action_one[:797] + "..."
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action_one} reward={reward:.2f} done={done_val} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    # Sample spec: rewards at 2 decimals; official sample uses score with 3 decimals
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def format_action_for_log(action_type: str, message: str) -> str:
    """Single-line action string similar to sample: type('message...')."""
    safe = message.replace("\\", "\\\\").replace("'", "\\'")
    safe = safe.replace("\n", " ").replace("\r", " ")
    if len(safe) > 400:
        safe = safe[:397] + "..."
    return f"{action_type}('{safe}')"


def run_episode(task_key: str, scenario_index: int = 0) -> dict[str, Any]:
    """One episode; stdout is only START / STEP / END."""
    task_name = TASK_NAMES.get(task_key, task_key)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_info: dict[str, Any] = {}

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_payload = {
            "task_type": task_key,
            "scenario_index": scenario_index,
            "seed": RESET_SEED,
        }
        resp = requests.post(f"{ENV_BASE_URL}/reset", json=reset_payload, timeout=60)
        if resp.status_code != 200:
            _stderr(f"reset failed: {resp.status_code} {resp.text}")
            return _finalize_episode(rewards, steps_taken, 0.0, False)

        observation = resp.json()
        conversation: List[str] = [f"Customer: {observation['customer_message']}"]
        done = False
        step_num = 0

        while not done:
            step_num += 1
            action = generate_action(observation, task_key, conversation)
            step_payload = {
                "message": action["message"],
                "action_type": action["action_type"],
                "confidence": action["confidence"],
            }
            action_str = format_action_for_log(action["action_type"], action["message"])

            err: Optional[str] = None
            try:
                sresp = requests.post(f"{ENV_BASE_URL}/step", json=step_payload, timeout=120)
                if sresp.status_code != 200:
                    err = sresp.text[:500]
                    log_step(step_num, action_str, 0.0, True, err)
                    rewards.append(0.0)
                    steps_taken = step_num
                    return _finalize_episode(rewards, steps_taken, 0.0, False)

                result = sresp.json()
                observation = result["observation"]
                reward_obj = result["reward"]
                r_val = float(reward_obj["value"])
                done = bool(result["done"])
                last_info = result.get("info") or {}

                log_step(step_num, action_str, r_val, done, None)
                rewards.append(r_val)
                steps_taken = step_num

                conversation.append(f"Agent: {action['message']}")
                conversation.append(f"Customer: {observation['customer_message']}")
            except Exception as exc:
                err = str(exc)[:500]
                log_step(step_num, action_str, 0.0, True, err)
                rewards.append(0.0)
                steps_taken = step_num
                return _finalize_episode(rewards, steps_taken, 0.0, False)

        tr = last_info.get("task_result") or {}
        raw_score = float(tr.get("score", 0.0))
        if not tr:
            raw_score = sum(rewards) / max(1, len(rewards)) if rewards else 0.0
        score = min(1.0, max(0.0, raw_score))
        success = score >= SUCCESS_SCORE_THRESHOLD
        return _finalize_episode(rewards, steps_taken, score, success)

    except Exception as e:
        _stderr(f"episode error: {e}")
        return _finalize_episode(rewards, steps_taken, 0.0, False)


def _finalize_episode(
    rewards: List[float],
    steps_taken: int,
    score: float,
    success: bool,
) -> dict[str, Any]:
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {
        "score": score,
        "steps": steps_taken,
        "success": success,
        "rewards": rewards,
    }


def generate_action(observation: dict, task_type: str, conversation_history: list[str]) -> dict:
    if USE_LLM and OPENAI_CLIENT is not None:
        return generate_llm_action(observation, task_type, conversation_history)
    return generate_rule_based_action(observation, task_type, conversation_history)


def generate_llm_action(observation: dict, task_type: str, conversation_history: list[str]) -> dict:
    system_prompt = build_system_prompt(observation, task_type)
    messages: List[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    for line in conversation_history:
        if line.startswith("Customer:"):
            messages.append({"role": "user", "content": line[len("Customer:") :].strip()})
        elif line.startswith("Agent:"):
            messages.append({"role": "assistant", "content": line[len("Agent:") :].strip()})

    messages.append({"role": "user", "content": observation.get("customer_message", "")})

    content = ""
    for attempt in range(2):
        try:
            response = OPENAI_CLIENT.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.7 if attempt == 0 else 0.3,
                max_tokens=500,
            )
            raw = response.choices[0].message.content
            content = raw.strip() if raw else ""
            if content:
                break
            _stderr(f"empty LLM response attempt {attempt + 1}")
        except Exception as e:
            _stderr(f"LLM error: {e}")
            content = get_rule_based_response(observation, task_type, conversation_history)
            break

    if not content:
        content = get_rule_based_response(observation, task_type, conversation_history)

    action_type = infer_action_type(content, task_type, conversation_history)
    return {"message": content, "action_type": action_type, "confidence": 0.7}


def generate_rule_based_action(observation: dict, task_type: str, conversation_history: list[str]) -> dict:
    content = get_rule_based_response(observation, task_type, conversation_history)
    action_type = infer_action_type(content, task_type, conversation_history)
    return {"message": content, "action_type": action_type, "confidence": 0.7}


def build_system_prompt(observation: dict, task_type: str) -> str:
    scenario = observation.get("scenario_description", "")
    prompts = {
        "easy": (
            "You are a friendly and helpful customer service representative for TechStore. "
            "Answer customer questions accurately and politely. Be concise and professional. "
            "Respond with plain text only — no markdown headers or emojis."
        ),
        "medium": (
            "You are an experienced customer service representative for TechStore. "
            "Handle complaints with empathy and concrete solutions. "
            "Plain text only, under 200 words."
        ),
        "hard": (
            "You are a senior representative for TechStore. De-escalate, gather facts, "
            "offer resolution or escalation. Plain text only, under 200 words."
        ),
    }
    base = prompts.get(task_type, prompts["easy"])
    if scenario:
        base += f"\n\nScenario: {scenario}"
    return base


def infer_action_type(message: str, task_type: str, conversation_history: list[str]) -> str:
    msg = message.lower()
    has_escalation = any(w in msg for w in ["escalate", "supervisor", "manager", "transfer"])
    has_closing = any(w in msg for w in ["anything else", "case closed", "have a great", "thank you for contacting"])
    has_apology = any(w in msg for w in ["i'm sorry", "i apologize", "i am sorry", "sincerely apologize"])
    has_solution = any(w in msg for w in ["replacement", "refund", "resolve", "fix", "process", "initiate", "arrange"])
    has_question = "?" in message

    if has_escalation:
        return "escalate"
    if has_closing and has_solution:
        return "close"
    if has_apology and has_solution:
        return "answer"
    if has_question and not has_apology and not has_solution:
        return "ask_clarify"
    if has_apology and not has_solution:
        return "acknowledge"
    return "answer"


def get_rule_based_response(observation: dict, task_type: str, conversation_history: list[str]) -> str:
    msg = observation.get("customer_message", "").lower()
    turn_count = len([l for l in conversation_history if l.startswith("Agent:")])

    if task_type == "easy":
        if any(w in msg for w in ["shipping", "delivery", "ship"]):
            return (
                "We offer Standard (5-7 business days, $5.99), Express (2-3 business days, $12.99), "
                "and Overnight (next business day, $24.99) shipping. Free standard shipping on orders over $50!"
            )
        if any(w in msg for w in ["return", "refund"]):
            return (
                "You can return items within 30 days of delivery in original condition. "
                "We provide free return shipping labels and process refunds within 5-7 business days."
            )
        if any(w in msg for w in ["payment", "pay", "credit"]):
            return (
                "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay. "
                "For orders over $100, we also offer Klarna installment payments."
            )
        if any(w in msg for w in ["track", "tracking"]):
            return (
                "You'll receive a tracking number via email within 24 hours of shipping. "
                "Use it on our website or the carrier's site to track your package."
            )
        if any(w in msg for w in ["warranty", "guarantee"]):
            return "All products include a 1-year manufacturer warranty. We also offer a 2-year extended warranty for $29.99."
        if any(w in msg for w in ["cancel", "cancellation"]):
            return "You can cancel your order within 1 hour of placement if it hasn't shipped yet. Call 1-800-555-0199 with your order number."
        if any(w in msg for w in ["hours", "contact", "support"]):
            return "Our team is available Monday-Friday 8AM-8PM EST and Saturday 9AM-5PM EST. Call 1-800-555-0199 or email support@techstore.com."
        if turn_count >= 1:
            return "Is there anything else I can help you with today?"
        return "Thank you for your question! Could you provide more details so I can assist you better?"

    if task_type == "medium":
        if turn_count == 0:
            if any(w in msg for w in ["defective", "broken", "doesn't work"]):
                return (
                    "I'm very sorry about the defective product. I can arrange a replacement with expedited shipping "
                    "at no cost, or a full refund, plus a prepaid return label. Which do you prefer?"
                )
            if any(w in msg for w in ["late", "delayed", "still haven't"]):
                return (
                    "I apologize for the delay. I'll check tracking and give an updated estimate "
                    "and refund your shipping for the inconvenience."
                )
            if any(w in msg for w in ["charged", "billing", "double"]):
                return (
                    "I'll look into the billing issue now. If a duplicate charge is confirmed, "
                    "I'll initiate an immediate refund and send you a reference number."
                )
            return "I'm here to help. Could you provide your order number?"
        if turn_count == 1:
            if any(w in msg for w in ["tell me more", "what about"]):
                return "Replacement ships express at no cost (2-3 business days), or refund in 5-7 business days. Which do you prefer?"
            if any(w in msg for w in ["order number", "order #"]):
                return "Thank you — I've located your order and will process the resolution now."
            return "I'm processing the resolution now; you'll get a confirmation email shortly."
        return "You're all set. Is there anything else I can help with today?"

    if turn_count == 0:
        return (
            "I sincerely apologize. I'm escalating to my supervisor and processing a full refund now. "
            "We'll investigate this with our supplier."
        )
    if turn_count == 1:
        if any(w in msg for w in ["not what i asked", "explained", "listening"]):
            return (
                "Your refund is processing; a supervisor will contact you within 24 hours. "
                "We're also reviewing this product line."
            )
        return (
            "A supervisor will contact you within 24 hours. Your refund is initiated. "
            "Is there anything else I can do today?"
        )
    return (
        "Your case is high priority; a supervisor will reach out. Your refund is in process. "
        "Thank you for your patience."
    )


def main() -> None:
    _stderr("Customer Service Bot inference (stdout = [START]/[STEP]/[END] only)")
    _stderr(f"ENV_BASE_URL={ENV_BASE_URL} API_BASE_URL={API_BASE_URL} MODEL_NAME={MODEL_NAME}")
    _stderr(f"USE_LLM={USE_LLM} BENCHMARK={BENCHMARK}")

    task_keys = ["easy", "medium", "hard"]
    results: List[dict[str, Any]] = []
    for key in task_keys:
        results.append(run_episode(key, scenario_index=0))

    avg = sum(r["score"] for r in results) / max(1, len(results))
    _stderr(f"Done. Per-task scores: {[round(r['score'], 4) for r in results]} avg={avg:.4f}")


if __name__ == "__main__":
    main()
