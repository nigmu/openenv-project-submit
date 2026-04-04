"""
train.py — Train a lightweight LLM on the Math Word Problem environment
           using GRPO (Group Relative Policy Optimization) via TRL.

Key fixes vs previous versions:
  1. task_level now stored in dataset → reward_fn reads from kwargs (not prompt length)
  2. task_level_id stored in dataset → reset() loads the EXACT problem the model saw
     (no more random.choice picking a different problem than the one being graded)

Set OPENENV_FAST_TRAIN=0 for a slower, heavier run.
"""

import atexit
import os
import re
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNTIME_PATH = os.path.join(BASE_DIR, "RUNTIME")
sys.path.insert(0, RUNTIME_PATH)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from demo.client import MathEnv
from demo.models import MathAction

ENV_URL = "http://localhost:8000"
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
FAST_TRAIN = os.environ.get("OPENENV_FAST_TRAIN", "1").lower() not in ("0", "false", "no")

# ── Problem bank ──────────────────────────────────────────────────────────────
# Each entry carries:
#   id          → matches TASK_BANK key in demo_environment.py
#   problem     → the text shown to the model
#   task_level  → "easy" / "medium" / "hard" (passed to env for logging)
# The id is what makes grading deterministic: reset(task_level_id=id) loads
# exactly this problem, not a random one from the same level.

ALL_PROBLEMS = [
    # easy
    {"id": "easy_0", "problem": "Sarah has 12 apples. She gives away 5. How many apples does she have left?",          "task_level": "easy"},
    {"id": "easy_1", "problem": "A shop sells 8 red pens and 6 blue pens. How many pens are there in total?",          "task_level": "easy"},
    {"id": "easy_2", "problem": "Tom walks 3 km to school and 3 km back home every day. How many km does he walk in a day?", "task_level": "easy"},
    # medium
    {"id": "medium_0", "problem": "A train travels at 60 km/h for 2 hours, then at 80 km/h for 1 hour. What is the total distance traveled in km?", "task_level": "medium"},
    {"id": "medium_1", "problem": "John earns $120 per day. He works for 5 days and then spends $200 on groceries. How many dollars does he have left?", "task_level": "medium"},
    {"id": "medium_2", "problem": "A rectangle has a length of 15 cm and a width of 8 cm. What is its perimeter in cm?", "task_level": "medium"},
    # hard
    {"id": "hard_0", "problem": "A store marks up its products by 40%, then offers a 15% discount. What is the final price in dollars of an item that originally costs $200?", "task_level": "hard"},
    {"id": "hard_1", "problem": "Three workers can build a wall in 12 days working together. How many days will it take 4 workers to build the same wall?", "task_level": "hard"},
    {"id": "hard_2", "problem": "A sum of money doubles in 8 years at simple interest. What is the annual interest rate as a percentage?", "task_level": "hard"},
]

# ── Prompt format ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a math problem solver. "
    "Think through the problem step by step, then write your final answer on the last line "
    "in this exact format:\nAnswer: <number>\n"
    "Only put the number after 'Answer:' — no units, no words."
)


def make_prompt(problem: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{problem}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def parse_answer(text: str) -> float:
    """
    Priority 1: explicit 'Answer: X' line.
    Priority 2: last number in the reasoning chain.
    """
    match = re.search(r"[Aa]nswer:\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(numbers[-1]) if numbers else 0.0


# ── Reward function ───────────────────────────────────────────────────────────

_reward_sync_env = None


def _get_reward_env():
    global _reward_sync_env
    if _reward_sync_env is None:
        client = MathEnv(base_url=ENV_URL).sync()
        client.connect()
        _reward_sync_env = client
    return _reward_sync_env


def _close_reward_env() -> None:
    global _reward_sync_env
    if _reward_sync_env is not None:
        try:
            _reward_sync_env.close()
        except Exception:
            pass
        _reward_sync_env = None


atexit.register(_close_reward_env)


def reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """
    Score each completion by calling the environment.

    kwargs contains the extra dataset columns GRPOTrainer passes through:
      - task_level_id : e.g. "easy_0" — tells reset() the exact problem to load
      - task_level    : e.g. "easy"   — for display only
    """
    global _reward_sync_env
    rewards = []

    # task_level_ids = kwargs.get("   ", ["easy_0"] * len(completions))
    task_level_ids = kwargs.get("task_level_id", ["easy_0"] * len(completions))
    problems = kwargs.get("problem", ["easy_0"] * len(completions))
    task_levels    = kwargs.get("task_level",    ["easy"]   * len(completions))

    for i, (completion, prompt, task_level_id, task_level, problem) in enumerate(
        zip(completions, prompts, task_level_ids, task_levels, problems)
    ):
        answer = parse_answer(completion)

        print(f"\n  [ENV call {i+1}]")
        print(f"  task_level_id  : {task_level_id}")
        # print(f"  task_level     : {task_level}")
        print(f"  problem        : {problem}")
        print(f"  model output   : {completion.strip()[:80]!r}")
        print(f"  parsed answer  : {answer}")

        try:
            env = _get_reward_env()

            # Pass the exact ID so the env loads the same problem the model saw
            print(f"  → reset(task_level_id={task_level_id!r})")
            env.reset(task_level_id=task_level_id)

            print(f"  → step(answer={answer})")
            step_result = env.step(MathAction(answer=answer, reasoning=completion))

            reward = step_result.reward if step_result.reward is not None else 0.0
            print(f"  ← reward: {reward}  ({'✓ correct' if reward == 1.0 else '✗'})")
            rewards.append(reward)

        except Exception as exc:
            print(f"  ✗ env error: {exc}")
            _reward_sync_env = None
            rewards.append(0.0)

    return rewards


# ── Dataset ───────────────────────────────────────────────────────────────────

def build_dataset(entries: list[dict]) -> list[dict]:
    """
    Each row has three columns:
      prompt        → formatted input the model sees
      task_level_id → e.g. "easy_0" — passed to reset() for exact problem lookup
      task_level    → e.g. "easy"   — passed through for logging
    """
    return [
        {
            "prompt":        make_prompt(e["problem"]),
            "task_level_id": e["id"],
            "task_level":    e["task_level"],
            "problem":       e["problem"],
        }
        for e in entries
    ]


# ── Hardware detection ────────────────────────────────────────────────────────

def _hardware_kwargs() -> dict:
    if os.environ.get("OPENENV_FORCE_CPU", "").lower() in ("1", "true", "yes"):
        return {"model_dtype": torch.float32, "use_cpu": True, "bf16": False, "fp16": False}
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        if torch.cuda.is_bf16_supported():
            return {"model_dtype": torch.bfloat16, "use_cpu": False, "bf16": True, "fp16": False}
        return {"model_dtype": torch.float16, "use_cpu": False, "bf16": False, "fp16": True}
    return {"model_dtype": torch.float32, "use_cpu": True, "bf16": False, "fp16": False}


def _log_device(hw: dict) -> None:
    if hw["use_cpu"]:
        print("Device: CPU (float32). Training will be slow — consider a free Colab GPU.")
    else:
        dtype = "bfloat16" if hw["bf16"] else "float16"
        print(f"Device: CUDA — {torch.cuda.get_device_name(0)} ({dtype})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    hw = _hardware_kwargs()
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=hw["model_dtype"])

    entries = ALL_PROBLEMS[:9] if FAST_TRAIN else ALL_PROBLEMS
    dataset = build_dataset(entries)

    if FAST_TRAIN:
        config = GRPOConfig(
            output_dir="./math_env_model",
            max_steps=8,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=5e-6,
            num_generations=2,
            max_completion_length=120,
            logging_steps=1,
            save_steps=500,
            report_to="none",
            use_cpu=hw["use_cpu"],
            bf16=hw["bf16"],
            fp16=hw["fp16"],
        )
        eval_max_new = 120
    else:
        config = GRPOConfig(
            output_dir="./math_env_model",
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=5e-6,
            num_generations=4,
            max_completion_length=200,
            logging_steps=5,
            save_steps=50,
            report_to="none",
            use_cpu=hw["use_cpu"],
            bf16=hw["bf16"],
            fp16=hw["fp16"],
        )
        eval_max_new = 150

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    mode = "FAST (set OPENENV_FAST_TRAIN=0 for full run)" if FAST_TRAIN else "FULL"
    print(f"\nStarting GRPO training [{mode}]...")
    _log_device(hw)
    print("(env server must be running: uvicorn server.app:app --port 8000)\n")

    try:
        trainer.train()
    finally:
        _close_reward_env()

    trainer.save_model("./math_env_model/final")
    print("\nTraining complete. Model saved to ./math_env_model/final")

    # ── Quick evaluation ──────────────────────────────────────────────────────
    print("\n── Quick evaluation (first 3 problems) ──")
    model.eval()
    device = next(model.parameters()).device

    for entry in ALL_PROBLEMS[:3]:
        prompt = make_prompt(entry["problem"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=eval_max_new,
                do_sample=False,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        answer = parse_answer(response)
        print(f"\nProblem    : {entry['problem']}")
        print(f"ID         : {entry['id']}  |  Level: {entry['task_level']}")
        print(f"Response   : {response.strip()}")
        print(f"Parsed     : {answer}")


if __name__ == "__main__":
    main()