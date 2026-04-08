#!/usr/bin/env python3
"""Comprehensive test suite for Customer Service Bot environment."""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from server.environment import CustomerServiceEnv
from models import Action, ActionType


def test_reset_all_tasks():
    """Test that reset() works for all task types."""
    print("=" * 60)
    print("TEST: Reset all task types")
    print("=" * 60)

    env = CustomerServiceEnv()
    for task_type in ["easy", "medium", "hard"]:
        obs = env.reset(task_type=task_type, scenario_index=0)
        assert obs.task_type == task_type, f"Expected task_type={task_type}, got {obs.task_type}"
        assert obs.turn_count == 0, f"Expected turn_count=0, got {obs.turn_count}"
        assert len(obs.customer_message) > 0, "Customer message should not be empty"
        print(f"  {task_type}: OK - '{obs.customer_message[:60]}...'")

    print("  PASSED\n")


def test_easy_task_episode():
    """Test a complete easy task episode."""
    print("=" * 60)
    print("TEST: Easy task full episode")
    print("=" * 60)

    env = CustomerServiceEnv(seed=42)
    obs = env.reset(task_type="easy", scenario_index=0)
    print(f"  Customer: {obs.customer_message}")

    responses = [
        (
            "We offer Standard (5-7 business days, $5.99), Express (2-3 days, $12.99), and Overnight (next day, $24.99). Free standard shipping on orders over $50!",
            "answer",
        ),
        ("Is there anything else I can help you with today?", "ask_clarify"),
    ]

    total_reward = 0.0
    for i, (msg, act_type) in enumerate(responses):
        action = Action(message=msg, action_type=ActionType(act_type))
        obs, reward, done, info = env.step(action)
        total_reward += reward.value
        tr = info.get("task_result", {})
        print(f"  Turn {i+1}: reward={reward.value:.3f}, task_score={tr.get('score', 0):.3f}, done={done}")
        assert 0.0 <= reward.value <= 1.0, f"Reward out of range: {reward.value}"
        assert 0.0 <= tr.get("score", 0) <= 1.0, f"Task score out of range: {tr.get('score')}"

    state = env.state()
    assert state.turn_count == 2, f"Expected 2 turns, got {state.turn_count}"
    assert state.task_type == "easy"
    assert len(state.conversation_history) >= 2

    print(f"  Total reward: {total_reward:.3f}")
    print(f"  Final mood: {state.customer_mood:.1f}")
    print("  PASSED\n")


def test_medium_task_episode():
    """Test a complete medium task episode."""
    print("=" * 60)
    print("TEST: Medium task full episode")
    print("=" * 60)

    env = CustomerServiceEnv(seed=42)
    obs = env.reset(task_type="medium", scenario_index=0)
    print(f"  Customer: {obs.customer_message}")

    responses = [
        (
            "I am very sorry to hear about the defective charger. I understand how frustrating this must be, especially when you need it for work. I can immediately arrange a replacement with expedited shipping at no cost, or offer a full refund. I will also provide a prepaid return shipping label for the defective unit.",
            "answer",
        ),
        (
            "I have initiated the replacement order for TS-48291. You will receive the tracking number within 24 hours. The replacement will arrive via express shipping at no extra cost.",
            "answer",
        ),
    ]

    total_reward = 0.0
    for i, (msg, act_type) in enumerate(responses):
        action = Action(message=msg, action_type=ActionType(act_type))
        obs, reward, done, info = env.step(action)
        total_reward += reward.value
        tr = info.get("task_result", {})
        print(f"  Turn {i+1}: reward={reward.value:.3f}, task_score={tr.get('score', 0):.3f}, done={done}")
        assert 0.0 <= reward.value <= 1.0, f"Reward out of range: {reward.value}"

    state = env.state()
    assert state.task_type == "medium"
    print(f"  Total reward: {total_reward:.3f}")
    print("  PASSED\n")


def test_hard_task_episode():
    """Test a complete hard task episode."""
    print("=" * 60)
    print("TEST: Hard task full episode")
    print("=" * 60)

    env = CustomerServiceEnv(seed=42)
    obs = env.reset(task_type="hard", scenario_index=0)
    print(f"  Customer: {obs.customer_message}")

    responses = [
        (
            "I sincerely apologize for this experience. I completely understand your frustration - receiving two defective products is absolutely unacceptable. I am escalating this to my supervisor immediately. In the meantime, I want to process a full refund for your order TS-46721 right now.",
            "escalate",
        ),
        (
            "My supervisor will contact you within 24 hours with a resolution. Your refund has been initiated. We are also investigating the quality issue with our supplier.",
            "answer",
        ),
    ]

    total_reward = 0.0
    for i, (msg, act_type) in enumerate(responses):
        action = Action(message=msg, action_type=ActionType(act_type))
        obs, reward, done, info = env.step(action)
        total_reward += reward.value
        tr = info.get("task_result", {})
        print(f"  Turn {i+1}: reward={reward.value:.3f}, task_score={tr.get('score', 0):.3f}, done={done}")
        assert 0.0 <= reward.value <= 1.0, f"Reward out of range: {reward.value}"

    state = env.state()
    assert state.task_type == "hard"
    print(f"  Total reward: {total_reward:.3f}")
    print("  PASSED\n")


def test_state_management():
    """Test state() returns correct data."""
    print("=" * 60)
    print("TEST: State management")
    print("=" * 60)

    env = CustomerServiceEnv(seed=42)
    obs = env.reset(task_type="easy", scenario_index=0)

    state = env.state()
    assert state.customer_mood == 7.0, f"Expected mood=7.0, got {state.customer_mood}"
    assert state.turn_count == 0
    assert state.resolution_status == "ongoing"
    assert state.episode_done == False
    assert state.task_type == "easy"
    print(f"  Initial state: mood={state.customer_mood}, turns={state.turn_count}, done={state.episode_done}")

    action = Action(message="Hello! How can I help?", action_type=ActionType.ANSWER)
    env.step(action)

    state = env.state()
    assert state.turn_count == 1
    assert len(state.conversation_history) >= 2
    assert len(state.satisfaction_trajectory) >= 2
    print(f"  After 1 step: mood={state.customer_mood:.1f}, turns={state.turn_count}")
    print("  PASSED\n")


def test_reward_range():
    """Test that all rewards are within [0.0, 1.0]."""
    print("=" * 60)
    print("TEST: Reward range validation")
    print("=" * 60)

    env = CustomerServiceEnv(seed=42)

    for task_type in ["easy", "medium", "hard"]:
        obs = env.reset(task_type=task_type, scenario_index=0)
        for _ in range(3):
            action = Action(message="Thank you for contacting us. How can I assist you today?", action_type=ActionType.ANSWER)
            obs, reward, done, info = env.step(action)
            assert 0.0 <= reward.value <= 1.0, f"Reward out of range for {task_type}: {reward.value}"
            if done:
                break
        print(f"  {task_type}: All rewards in [0.0, 1.0] - OK")

    print("  PASSED\n")


def test_deterministic_scoring():
    """Test that scoring is deterministic with same seed."""
    print("=" * 60)
    print("TEST: Deterministic scoring")
    print("=" * 60)

    results = []
    for run in range(2):
        env = CustomerServiceEnv(seed=42)
        obs = env.reset(task_type="easy", scenario_index=0)
        action = Action(message="We offer Standard (5-7 business days, $5.99), Express (2-3 days, $12.99), and Overnight (next day, $24.99). Free standard shipping on orders over $50!", action_type=ActionType.ANSWER)
        obs, reward, done, info = env.step(action)
        results.append({
            "reward": reward.value,
            "score": info["task_result"]["score"],
            "mood": info["customer_mood"],
        })

    assert results[0]["reward"] == results[1]["reward"], f"Rewards differ: {results[0]} vs {results[1]}"
    assert results[0]["score"] == results[1]["score"], f"Scores differ: {results[0]} vs {results[1]}"
    assert results[0]["mood"] == results[1]["mood"], f"Moods differ: {results[0]} vs {results[1]}"
    print(f"  Run 1: reward={results[0]['reward']:.3f}, score={results[0]['score']:.3f}")
    print(f"  Run 2: reward={results[1]['reward']:.3f}, score={results[1]['score']:.3f}")
    print("  PASSED\n")


def test_episode_boundaries():
    """Test that episodes end properly."""
    print("=" * 60)
    print("TEST: Episode boundaries")
    print("=" * 60)

    env = CustomerServiceEnv(seed=42)
    obs = env.reset(task_type="easy", scenario_index=0)

    done = False
    turns = 0
    while not done and turns < 20:
        action = Action(message="Help", action_type=ActionType.ANSWER)
        obs, reward, done, info = env.step(action)
        turns += 1

    assert done, f"Episode should have ended after {turns} turns"
    assert turns <= 5, f"Easy task should end within 5 turns, got {turns}"
    print(f"  Episode ended after {turns} turns (max for easy: 5)")
    print("  PASSED\n")


def main():
    print("\n" + "=" * 60)
    print("CUSTOMER SERVICE BOT - TEST SUITE")
    print("=" * 60 + "\n")

    tests = [
        test_reset_all_tasks,
        test_easy_task_episode,
        test_medium_task_episode,
        test_hard_task_episode,
        test_state_management,
        test_reward_range,
        test_deterministic_scoring,
        test_episode_boundaries,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}\n")
            failed += 1

    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
