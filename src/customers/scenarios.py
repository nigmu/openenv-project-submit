from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CustomerProfile:
    name: str
    mood: float  # 0-10
    patience: float  # 0-10
    issue_type: str
    urgency: int  # 1-5
    scenario_id: str
    scenario_description: str
    initial_message: str
    expected_resolution: str
    key_facts: list[str] = field(default_factory=list)
    escalation_triggers: list[str] = field(default_factory=list)


def _load_json(filename: str) -> dict:
    base = os.path.dirname(__file__)
    path = os.path.join(base, "..", "knowledge_base", filename)
    with open(path, "r") as f:
        return json.load(f)


SCENARIOS = {
    "easy": [
        CustomerProfile(
            name="Alice Chen",
            mood=7.0,
            patience=8.0,
            issue_type="shipping_inquiry",
            urgency=2,
            scenario_id="faq_shipping_time",
            scenario_description="Customer wants to know shipping timeframes",
            initial_message="Hi! I'm thinking of placing an order. How long does shipping usually take?",
            expected_resolution="Provide shipping options with timeframes and costs",
            key_facts=["standard", "5-7", "5.99", "express", "2-3", "12.99", "overnight", "next", "24.99", "free", "50"],
            escalation_triggers=[],
        ),
        CustomerProfile(
            name="Bob Martinez",
            mood=6.0,
            patience=7.0,
            issue_type="return_policy_inquiry",
            urgency=1,
            scenario_id="faq_return_policy",
            scenario_description="Customer asking about return policy before purchase",
            initial_message="Hello, I'd like to know what your return policy is before I buy something.",
            expected_resolution="Explain 30-day return window, condition requirements, and free return shipping",
            key_facts=["30 days", "unused", "original packaging", "free return", "refund", "5-7"],
            escalation_triggers=[],
        ),
        CustomerProfile(
            name="Carol White",
            mood=8.0,
            patience=9.0,
            issue_type="payment_methods",
            urgency=1,
            scenario_id="faq_payment",
            scenario_description="Customer wants to know accepted payment methods",
            initial_message="Hey there! What payment methods do you guys accept?",
            expected_resolution="List all accepted payment methods and mention Klarna installments",
            key_facts=["Visa", "Mastercard", "Amex", "PayPal", "Apple Pay", "Klarna", "100"],
            escalation_triggers=[],
        ),
        CustomerProfile(
            name="David Kim",
            mood=7.5,
            patience=8.0,
            issue_type="track_order",
            urgency=3,
            scenario_id="faq_tracking",
            scenario_description="Customer wants to track their order",
            initial_message="Hi, I placed an order 2 days ago and haven't received a tracking number yet. How can I track it?",
            expected_resolution="Explain tracking number email process and 24-hour window",
            key_facts=["tracking", "email", "24 hours", "website", "carrier"],
            escalation_triggers=["order placed more than 48 hours ago without tracking"],
        ),
        CustomerProfile(
            name="Emma Thompson",
            mood=6.5,
            patience=7.0,
            issue_type="warranty_inquiry",
            urgency=1,
            scenario_id="faq_warranty",
            scenario_description="Customer asking about product warranty",
            initial_message="Hello, do your products come with any kind of warranty?",
            expected_resolution="Explain 1-year standard warranty and 2-year extended option",
            key_facts=["1 year", "manufacturer", "defects", "2-year", "extended", "29.99"],
            escalation_triggers=[],
        ),
    ],
    "medium": [
        CustomerProfile(
            name="Frank Johnson",
            mood=3.0,
            patience=5.0,
            issue_type="defective_product",
            urgency=4,
            scenario_id="complaint_defective",
            scenario_description="Customer received a defective laptop charger",
            initial_message="I just received my laptop charger order and it doesn't work at all. I'm really frustrated because I needed this for work tomorrow. Order #TS-48291.",
            expected_resolution="Apologize, offer immediate replacement with expedited shipping, provide return label",
            key_facts=["TS-48291", "laptop charger", "defective", "work", "tomorrow"],
            escalation_triggers=["replacement also doesn't work", "demands compensation beyond replacement"],
        ),
        CustomerProfile(
            name="Grace Lee",
            mood=2.5,
            patience=4.0,
            issue_type="late_delivery",
            urgency=4,
            scenario_id="complaint_late",
            scenario_description="Customer's order is 5 days late",
            initial_message="This is unacceptable! I ordered a birthday gift 2 weeks ago with express shipping and it STILL hasn't arrived. The birthday is THIS WEEKEND. Order #TS-47832.",
            expected_resolution="Apologize sincerely, check tracking, offer shipping refund, expedite or replace",
            key_facts=["TS-47832", "express", "2 weeks", "birthday", "late"],
            escalation_triggers=["package confirmed lost", "demands full refund AND replacement"],
        ),
        CustomerProfile(
            name="Henry Davis",
            mood=3.5,
            patience=5.0,
            issue_type="billing_dispute",
            urgency=3,
            scenario_id="complaint_billing",
            scenario_description="Customer was charged twice for the same order",
            initial_message="I just noticed on my credit card statement that I was charged twice for order #TS-49103. I only placed one order! Can you fix this? That's $89.98 that shouldn't have been charged.",
            expected_resolution="Verify duplicate charge, apologize, initiate refund, provide reference number",
            key_facts=["TS-49103", "charged twice", "89.98", "credit card"],
            escalation_triggers=["charge over $500", "requests supervisor"],
        ),
    ],
    "hard": [
        CustomerProfile(
            name="Isabella Rodriguez",
            mood=1.5,
            patience=2.0,
            issue_type="angry_customer_defective",
            urgency=5,
            scenario_id="escalation_angry_defective",
            scenario_description="Customer's second replacement is also defective, very angry",
            initial_message="THIS IS THE SECOND TIME I've received a defective product from you people! I ordered a wireless mouse, the first one was broken, you sent a replacement, and THAT one is broken too! I want a FULL REFUND and I want to speak to your MANAGER. Order #TS-46721. This is absolutely ridiculous!",
            expected_resolution="De-escalate, apologize profusely, offer full refund immediately, escalate to supervisor, investigate quality issue",
            key_facts=["TS-46721", "wireless mouse", "second", "defective", "refund", "manager"],
            escalation_triggers=["demands manager (should escalate)", "threatens legal action", "threatens social media"],
        ),
        CustomerProfile(
            name="James Wilson",
            mood=2.0,
            patience=3.0,
            issue_type="late_delivery_angry",
            urgency=5,
            scenario_id="escalation_late_angry",
            scenario_description="Customer's business order is very late, causing business impact",
            initial_message="Your company has completely messed up our business order. We ordered 50 laptops for our new office opening on Monday. It's now Friday and we only received 20 of them. The other 30 are MIA. This is costing us THOUSANDS in delayed operations. Order #TS-45890. I need this resolved NOW or we're switching vendors permanently.",
            expected_resolution="Acknowledge severity, check bulk order status, offer immediate solutions, escalate to account manager, provide compensation",
            key_facts=["TS-45890", "50", "laptops", "20", "30", "business", "Monday"],
            escalation_triggers=["demands account manager (should escalate)", "threatens to switch vendors", "mentions financial damages"],
        ),
    ],
}


def get_scenario(difficulty: str, scenario_index: int = 0) -> CustomerProfile:
    scenarios = SCENARIOS.get(difficulty, [])
    if not scenarios:
        raise ValueError(f"No scenarios found for difficulty: {difficulty}")
    idx = scenario_index % len(scenarios)
    return scenarios[idx]


def get_all_scenarios() -> dict[str, list[CustomerProfile]]:
    return SCENARIOS
