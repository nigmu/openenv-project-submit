from __future__ import annotations

import re
from .scoring import SENTIMENT_WORDS


def analyze_sentiment(text: str) -> float:
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)

    score = 0.0
    count = 0

    for word in words:
        if word in SENTIMENT_WORDS["positive"]:
            score += SENTIMENT_WORDS["positive"][word]
            count += 1
        elif word in SENTIMENT_WORDS["negative"]:
            score += SENTIMENT_WORDS["negative"][word]
            count += 1

    if count == 0:
        return 0.0

    return max(-1.0, min(1.0, score / count))


def sentiment_to_mood_delta(sentiment: float) -> float:
    return sentiment * 1.5
