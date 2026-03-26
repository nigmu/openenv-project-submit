import requests

BASE_URL = "http://127.0.0.1:8000"

def run_task(level):
    # Reset env
    obs = requests.get(f"{BASE_URL}/reset").json()

    # Dummy agent action
    action = {
        "action_type": "test",
        "payload": {}
    }

    # Step
    response = requests.post(f"{BASE_URL}/step", json=action)
    
    # Get score
    grader_input = {
        "action": action,
        "level": level
    }

    score = requests.post(f"{BASE_URL}/grader", json=grader_input).json()

    return score


def run_baseline():
    results = {}
    for level in ["easy", "medium", "hard"]:
        results[level] = run_task(level)

    return results


if __name__ == "__main__":
    print(run_baseline())