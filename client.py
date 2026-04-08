from __future__ import annotations

import os
from openai import OpenAI
from typing import Optional


class ServiceBotClient:
    def __init__(
        self,
        api_base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.api_base_url = api_base_url or os.getenv(
            "API_BASE_URL", os.getenv("LLM_API_BASE", "http://127.0.0.1:11434/v1")
        )
        self.model_name = model_name or os.getenv("MODEL_NAME", os.getenv("OLLAMA_MODEL", "llama3.2"))
        self.api_key = api_key or (
            os.getenv("HF_TOKEN")
            or os.getenv("OLLAMA_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or "ollama"
        )

        self.client = OpenAI(
            base_url=self.api_base_url,
            api_key=self.api_key,
        )

    def generate_response(
        self,
        observation: dict,
        system_prompt: Optional[str] = None,
    ) -> dict:
        if system_prompt is None:
            system_prompt = self._build_system_prompt(observation)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": observation.get("customer_message", "")},
        ]

        if observation.get("conversation_history"):
            history = observation["conversation_history"]
            for i in range(0, len(history) - 1, 2):
                if i + 1 < len(history):
                    messages.insert(1, {"role": "user", "content": history[i]})
                    messages.insert(2, {"role": "assistant", "content": history[i + 1]})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=300,
        )

        content = response.choices[0].message.content
        action_type = self._infer_action_type(content, observation)

        return {
            "message": content,
            "action_type": action_type,
            "confidence": 0.7,
        }

    def _build_system_prompt(self, observation: dict) -> str:
        task_type = observation.get("task_type", "easy")
        scenario = observation.get("scenario_description", "")

        prompts = {
            "easy": (
                "You are a friendly and helpful customer service representative for TechStore. "
                "Answer customer questions accurately using the company knowledge base. "
                "Be polite, concise, and professional. "
                "Only provide information you are certain about."
            ),
            "medium": (
                "You are an experienced customer service representative for TechStore. "
                "A customer has a complaint that needs resolution. "
                "Show empathy, acknowledge their problem, identify the issue accurately, "
                "and propose a concrete solution. Follow company procedures."
            ),
            "hard": (
                "You are a senior customer service representative for TechStore. "
                "A customer is very upset and may need escalation. "
                "Your priorities: 1) De-escalate the situation, 2) Gather all relevant information, "
                "3) Provide a clear resolution path, 4) Escalate to a supervisor when appropriate. "
                "Stay calm and professional at all times."
            ),
        }

        base_prompt = prompts.get(task_type, prompts["easy"])

        if scenario:
            base_prompt += f"\n\nScenario context: {scenario}"

        base_prompt += (
            "\n\nRespond with a direct message to the customer. "
            "Be empathetic, professional, and solution-oriented."
        )

        return base_prompt

    def _infer_action_type(self, message: str, observation: dict) -> str:
        message_lower = message.lower()

        if any(word in message_lower for word in ["escalate", "supervisor", "manager", "transfer"]):
            return "escalate"
        elif any(word in message_lower for word in ["thank you", "is there anything else", "resolved", "close"]):
            return "close"
        elif "?" in message and message.count("?") >= 1:
            return "ask_clarify"
        elif any(word in message_lower for word in ["understand", "i see", "i hear", "acknowledged"]):
            return "acknowledge"
        else:
            return "answer"
