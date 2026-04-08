from __future__ import annotations

import re
from typing import Optional


STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "don", "now", "i", "me", "my", "myself", "we", "our", "ours", "you",
    "your", "he", "him", "his", "she", "her", "it", "its", "they", "them",
    "their", "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "and", "but", "if", "or", "because", "about", "against", "while",
}

SENTIMENT_WORDS = {
    "positive": {
        "thank": 0.8, "thanks": 0.8, "great": 0.7, "excellent": 0.9,
        "good": 0.6, "helpful": 0.7, "appreciate": 0.8, "happy": 0.8,
        "wonderful": 0.9, "perfect": 0.9, "awesome": 0.8, "nice": 0.6,
        "pleased": 0.7, "satisfied": 0.7, "resolved": 0.8, "okay": 0.4,
    },
    "negative": {
        "terrible": -0.9, "awful": -0.8, "worst": -0.9, "horrible": -0.9,
        "angry": -0.7, "frustrated": -0.6, "disappointed": -0.7, "unacceptable": -0.8,
        "ridiculous": -0.8, "useless": -0.7, "broken": -0.5, "refund": -0.3,
        "complaint": -0.5, "issue": -0.3, "problem": -0.3, "wrong": -0.4,
        "bad": -0.6, "poor": -0.5, "hate": -0.8, "furious": -0.9,
    },
}

PROFESSIONAL_PHRASES = [
    "i understand", "i apologize", "i'm sorry", "let me help", "i can assist",
    "thank you for", "i appreciate", "please allow me", "certainly", "absolutely",
    "i'd be happy to", "of course", "rest assured", "we value", "our team",
    "please don't hesitate", "feel free", "is there anything else", "how can i",
]

UNPROFESSIONAL_PHRASES = [
    "i don't know", "not my problem", "can't help", "that's not", "you should",
    "you need to", "obviously", "clearly", "as i said", "read the",
    "calm down", "relax", "whatever", "fine", "whatever you want",
]


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


def detect_intent(text: str) -> str:
    text_lower = text.lower()

    intent_map = {
        "shipping_inquiry": ["shipping", "delivery", "ship", "deliver", "arrive", "tracking"],
        "return_inquiry": ["return", "refund", "send back", "exchange", "money back"],
        "payment_inquiry": ["payment", "pay", "credit card", "paypal", "apple pay"],
        "warranty_inquiry": ["warranty", "guarantee", "defect", "coverage"],
        "contact_inquiry": ["hours", "contact", "support", "phone", "email"],
        "complaint_defective": ["defective", "broken", "doesn't work", "not working", "damaged"],
        "complaint_late": ["late", "delayed", "still haven't", "waiting", "overdue"],
        "complaint_billing": ["charged twice", "double charge", "billing", "overcharged", "incorrect charge"],
        "escalation": ["manager", "supervisor", "escalate", "speak to", "complaint"],
        "general_inquiry": ["question", "information", "tell me", "what is", "how do"],
    }

    best_intent = "general_inquiry"
    best_score = 0

    for intent, keywords in intent_map.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_score:
            best_score = score
            best_intent = intent

    return best_intent


def check_keyword_coverage(text: str, required_keywords: list[str]) -> float:
    if not required_keywords:
        return 1.0

    text_lower = text.lower()
    matched = 0

    for kw in required_keywords:
        if kw.lower() in text_lower:
            matched += 1

    return matched / len(required_keywords)


def assess_professionalism(text: str) -> float:
    text_lower = text.lower()

    professional_count = sum(1 for phrase in PROFESSIONAL_PHRASES if phrase in text_lower)
    unprofessional_count = sum(1 for phrase in UNPROFESSIONAL_PHRASES if phrase in text_lower)

    if professional_count == 0 and unprofessional_count == 0:
        return 0.5

    total = professional_count + unprofessional_count
    return max(0.0, min(1.0, professional_count / total))


def assess_empathy(text: str) -> float:
    text_lower = text.lower()

    empathy_indicators = [
        "i understand", "i'm sorry", "i apologize", "frustrating", "inconvenience",
        "upsetting", "disappointing", "i can imagine", "that must be", "feel",
        "empathize", "concern", "care about", "we value you", "your experience",
    ]

    matched = sum(1 for indicator in empathy_indicators if indicator in text_lower)
    return min(1.0, matched / 3.0)


def assess_conciseness(text: str, optimal_range: tuple[int, int] = (50, 300)) -> float:
    length = len(text)
    low, high = optimal_range

    if low <= length <= high:
        return 1.0
    elif length < low:
        return max(0.0, length / low)
    else:
        return max(0.0, 1.0 - (length - high) / (high * 2))


def detect_hallucination(text: str, known_facts: list[str]) -> float:
    text_lower = text.lower()

    specific_claims = re.findall(r'\$[\d,]+\.?\d*|\d+\s*(?:days?|hours?|weeks?|months?)', text_lower)

    if not specific_claims:
        return 1.0

    verified = 0
    for claim in specific_claims:
        claim_clean = claim.replace('$', '').replace(',', '').strip()
        claim_num = re.search(r'[\d.]+', claim_clean)
        claim_value = float(claim_num.group()) if claim_num else None

        for fact in known_facts:
            fact_lower = fact.lower()
            if claim in fact_lower:
                verified += 1
                break

            if claim_value is not None:
                fact_numbers = re.findall(r'[\d.]+', fact_lower)
                for fact_num_str in fact_numbers:
                    fact_num = float(fact_num_str)
                    if abs(fact_num - claim_value) < 0.1:
                        verified += 1
                        break
                else:
                    continue
                break

            if any(part in fact_lower for part in claim.split()):
                verified += 1
                break

    return verified / len(specific_claims) if specific_claims else 1.0
