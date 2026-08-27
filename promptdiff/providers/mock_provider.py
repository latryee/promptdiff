"""Deterministic Offline Mock LLM Provider.

Generates realistic text/JSON outputs, token counts, and latency deltas for zero-cost
local development, test suites, and CI/CD demonstration.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from typing import Any, Optional
from promptdiff.providers.base import BaseLLMProvider, ProviderResponse


class MockProvider(BaseLLMProvider):
    """Mock Provider that deterministically responds based on prompt contents."""

    def __init__(self, model_name: str = "mock-gpt-4o", simulate_delay: bool = True, **kwargs: Any):
        super().__init__(model_name, **kwargs)
        self.simulate_delay = simulate_delay

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = 2048,
    ) -> ProviderResponse:
        start_time = time.perf_counter()

        # Seed deterministic pseudo-random generator with prompt hash
        h = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), 16)
        base_latency = 150 + (h % 200)  # 150ms - 350ms

        if self.simulate_delay:
            # Quick micro-sleep to simulate async network flight time
            await asyncio.sleep(min(base_latency / 1000.0, 0.05))

        prompt_lower = prompt.lower()
        output_text = self._synthesize_output(prompt, prompt_lower, h)

        # Approximate token count (1 token ~= 4 chars)
        prompt_tokens = max(10, len(prompt) // 4)
        completion_tokens = max(5, len(output_text) // 4)
        total_tokens = prompt_tokens + completion_tokens

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        # If simulated sleep was capped, return realistic reported latency
        reported_latency = base_latency + (elapsed_ms % 20)

        return ProviderResponse(
            output=output_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=round(reported_latency, 2),
            model=self.model_name,
            raw_response={"mock": True, "hash": h},
        )

    def _synthesize_output(self, raw_prompt: str, prompt_lower: str, seed: int) -> str:
        """Synthesize context-aware output based on prompt cues."""

        # 1. JSON Extraction / Schema Scenarios
        if "json" in prompt_lower or "schema" in prompt_lower or "structured" in prompt_lower:
            # Check for customer support / ticket classification
            if "sentiment" in prompt_lower or "ticket" in prompt_lower or "support" in prompt_lower:
                if "v2" in prompt_lower or "urgency" in prompt_lower or "confidence" in prompt_lower:
                    # Version 2 output: Richer schema
                    return json.dumps(
                        {
                            "category": "technical_support",
                            "sentiment": "negative",
                            "urgency": "high",
                            "confidence": 0.94,
                            "summary": "Customer unable to complete checkout due to billing error 402.",
                            "action_required": True,
                        },
                        indent=2,
                    )
                else:
                    # Version 1 output: Basic schema
                    return json.dumps(
                        {
                            "category": "support",
                            "sentiment": "frustrated",
                            "summary": "Checkout payment issue",
                        },
                        indent=2,
                    )

            # Generic JSON extractor
            return json.dumps(
                {
                    "status": "success",
                    "extracted_entities": ["Acme Corp", "Invoice #4829", "$1,450.00"],
                    "confidence_score": 0.98,
                    "processed_at": "2026-08-27T12:00:00Z",
                },
                indent=2,
            )

        # 2. Customer Support Chatbot / Agent Scenarios
        if "support" in prompt_lower or "assistant" in prompt_lower or "customer" in prompt_lower:
            if "concise" in prompt_lower or "bullet" in prompt_lower or "v2" in prompt_lower:
                return (
                    "Hello! Thanks for reaching out.\n\n"
                    "Here is what I can help you with:\n"
                    "- Resetting your API credentials\n"
                    "- Upgrading subscription tier\n"
                    "- Reviewing invoice line items\n\n"
                    "Please reply with your preferred action."
                )
            else:
                return (
                    "Dear customer, thank you for contacting our 24/7 technical support hotline. "
                    "We are delighted to assist you with any inquiries you might have regarding your account, "
                    "billing settings, or platform configurations. How may I be of service to you today?"
                )

        # 3. Summarization Scenarios
        if "summar" in prompt_lower or "tldr" in prompt_lower:
            if "bullet" in prompt_lower:
                return (
                    "Key Takeaways:\n"
                    "* Q3 revenue grew by 24% YoY to $42.5M.\n"
                    "* Gross margins expanded to 78.4%.\n"
                    "* Enterprise customer count surpassed 1,200 accounts."
                )
            return "In Q3, the company generated $42.5M in revenue (24% YoY growth) with expanding 78.4% gross margins."

        # 4. Code / SQL Generation Scenarios
        if "sql" in prompt_lower or "query" in prompt_lower:
            return "SELECT user_id, COUNT(*) as order_count, SUM(total_amount) as lifetime_spend FROM orders WHERE created_at >= '2026-01-01' GROUP BY user_id ORDER BY lifetime_spend DESC LIMIT 10;"

        # 5. Default generic deterministic text
        return f"Deterministic mock response generated for prompt hash ({hex(seed)[2:10]}). All instructions respected."
