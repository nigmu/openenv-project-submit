"""
Math Word Problem Environment.

A real-world RL environment where an AI agent solves math word problems
across three difficulty levels. Each episode = one problem to solve.

Reward function gives partial credit:
  - 1.0  → exact answer
  - 0.8  → within 1% of correct answer
  - 0.4  → within 10% of correct answer
  - 0.0  → more than 10% off


Key design: TASK_BANK is a flat dict keyed by "level_index" (e.g. "easy_0").
reset() accepts a task_level_id and does a direct O(1) lookup — no random
selection, so the trainer always grades the problem it actually showed the model.
"""

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from ..models import MathAction, MathObservation


# ── Problem bank ──────────────────────────────────────────────────────────────
# Flat dict: key = "level_index", value = {problem, answer, level}
# This lets reward_fn pass a specific ID and get an exact match instantly.
# ─────────────────────────────────────────────────────────────────────────────

TASK_BANK = {
    # easy
    "easy_0": {"problem": "Sarah has 12 apples. She gives away 5. How many apples does she have left?",         "answer": 7.0,   "level": "easy"},
    "easy_1": {"problem": "A shop sells 8 red pens and 6 blue pens. How many pens are there in total?",         "answer": 14.0,  "level": "easy"},
    "easy_2": {"problem": "Tom walks 3 km to school and 3 km back home every day. How many km does he walk in a day?", "answer": 6.0,  "level": "easy"},
    "easy_3": {"problem": "There are 24 students in a class. 9 are absent today. How many students are present?","answer": 15.0,  "level": "easy"},
    "easy_4": {"problem": "A box has 5 rows of chocolates with 6 in each row. How many chocolates are there in total?", "answer": 30.0, "level": "easy"},
    "easy_5": {"problem": "A car travels 280 km on 40 liters of petrol. How many km per liter does it get?",    "answer": 7.0,   "level": "easy"},
    "easy_6": {"problem": "A farmer has 48 eggs and packs them into boxes of 6. How many boxes does he fill?",  "answer": 8.0,   "level": "easy"},

    # medium
    "medium_0": {"problem": "A train travels at 60 km/h for 2 hours, then at 80 km/h for 1 hour. What is the total distance traveled in km?", "answer": 200.0, "level": "medium"},
    "medium_1": {"problem": "John earns $120 per day. He works for 5 days and then spends $200 on groceries. How many dollars does he have left?", "answer": 400.0, "level": "medium"},
    "medium_2": {"problem": "A rectangle has a length of 15 cm and a width of 8 cm. What is its perimeter in cm?", "answer": 46.0, "level": "medium"},
    "medium_3": {"problem": "A water tank can hold 500 liters. It is currently 40% full. How many more liters are needed to completely fill it?", "answer": 300.0, "level": "medium"},
    "medium_4": {"problem": "Maria reads 25 pages each day. She wants to finish a 325-page book. How many days will it take her?", "answer": 13.0, "level": "medium"},
    "medium_5": {"problem": "A recipe uses 3 cups of flour to make 12 cookies. How many cups of flour are needed to make 60 cookies?", "answer": 15.0, "level": "medium"},

    # hard
    "hard_0": {"problem": "A store marks up its products by 40%, then offers a 15% discount. What is the final price in dollars of an item that originally costs $200?", "answer": 238.0, "level": "hard"},
    "hard_1": {"problem": "Three workers can build a wall in 12 days working together. How many days will it take 4 workers to build the same wall?", "answer": 9.0, "level": "hard"},
    "hard_2": {"problem": "A sum of money doubles in 8 years at simple interest. What is the annual interest rate as a percentage?", "answer": 12.5, "level": "hard"},
    "hard_3": {"problem": "A mixture of 40 liters is 25% alcohol. How many liters of pure alcohol must be added to make it 40% alcohol?", "answer": 10.0, "level": "hard"},
    "hard_4": {"problem": "Train A leaves a station at 9:00 AM traveling at 60 km/h. Train B leaves the same station at 10:00 AM in the same direction at 90 km/h. How many km from the station do they meet?", "answer": 180.0, "level": "hard"},
    "hard_5": {"problem": "A shopkeeper buys 100 kg of rice at $0.80/kg and 50 kg at $0.90/kg. He mixes them and sells at $1.00/kg. What is his total profit in dollars?", "answer": 75.0, "level": "hard"},
}

# Keys grouped by level — used when a random problem is needed (e.g. inference.py)
EASY_IDS   = [k for k in TASK_BANK if k.startswith("easy")]
MEDIUM_IDS = [k for k in TASK_BANK if k.startswith("medium")]
HARD_IDS   = [k for k in TASK_BANK if k.startswith("hard")]
LEVEL_IDS  = {"easy": EASY_IDS, "medium": MEDIUM_IDS, "hard": HARD_IDS}


class MathEnvironment(Environment):
    """
    Math Word Problem environment for RL training.

    Training flow (deterministic — used by train.py):
        env.reset(task_level_id="easy_0")   # pin to a specific problem
        env.step(MathAction(answer=7.0))    # grade that exact problem

    Inference / random flow:
        env.reset()                          # picks a random easy problem
        env.step(MathAction(answer=...))
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task: dict = {}
        self._task_level_id: str = "easy_0"

    def reset(self, task_level_id: str = None, task_level: str = "easy") -> MathObservation:
        """
        Start a new episode.

        Args:
            task_level_id : Specific problem ID e.g. "easy_0". When provided,
                            that exact problem is loaded — used by the trainer
                            so the model is always graded on what it was shown.
            task_level    : Fallback level ("easy"/"medium"/"hard") used when
                            task_level_id is not given — picks a random problem
                            from that level. Used by inference.py.

        Returns:
            MathObservation with the problem to solve (answer not revealed yet)
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)

        if task_level_id and task_level_id in TASK_BANK:
            # Deterministic: trainer pinned a specific problem
            self._task_level_id = task_level_id
            self._current_task = TASK_BANK[task_level_id]          # ← direct lookup, no random
        else:
            # Random fallback: pick any problem from the requested level
            level = task_level if task_level in LEVEL_IDS else "easy"
            self._task_level_id = random.choice(LEVEL_IDS[level])
            self._current_task = TASK_BANK[self._task_level_id]

        print(f"Reset is called")
        print(f"Resetting environment with task_level_id: {self._task_level_id}")

        return MathObservation(
            problem=self._current_task["problem"],
            task_level=self._current_task["level"],
            done=False,
            reward=0.0,
        )

    def step(self, action: MathAction) -> MathObservation:  # type: ignore[override]
        """
        Evaluate the agent's answer against the current task.

        Partial reward:
          1.0 → exact, 0.8 → within 1%, 0.4 → within 10%, 0.0 → otherwise
        """
        self._state.step_count += 1
        correct   = self._current_task["answer"]
        submitted = action.answer

        denom          = abs(correct) if abs(correct) > 1e-9 else 1.0
        relative_error = abs(submitted - correct) / denom

        if relative_error == 0:
            reward, is_correct = 1.0, True
            feedback = f"Correct! The answer is {correct}."
        elif relative_error <= 0.01:
            reward, is_correct = 0.8, False
            feedback = f"Very close! Correct answer is {correct}, you submitted {submitted}."
        elif relative_error <= 0.10:
            reward, is_correct = 0.4, False
            feedback = f"Partially correct. The answer is {correct}, you submitted {submitted}."
        else:
            reward, is_correct = 0.0, False
            feedback = f"Incorrect. The correct answer is {correct}, you submitted {submitted}."

        print(f"Step is called")
        print(f"Step result: {reward}, {is_correct}, {feedback}")

        return MathObservation(
            problem=self._current_task["problem"],
            task_level=self._current_task["level"],
            correct_answer=correct,
            is_correct=is_correct,
            feedback=feedback,
            done=True,
            reward=reward,
            metadata={
                "submitted_answer": submitted,
                "correct_answer": correct,
                "relative_error": relative_error,
                "step": self._state.step_count,
            },
        )

    @property
    def state(self) -> State:
        print(f"State is called")
        print(f"State: {self._state}")
        return self._state