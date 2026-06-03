"""Engagement agent: suggests study timing based on workload signals."""
from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()

try:
    from openai import AzureOpenAI
except Exception:  # pragma: no cover - runtime dependency
    AzureOpenAI = None

_CLIENT: Optional[Any] = None
_DEPLOYMENT: Optional[str] = None
if AzureOpenAI is not None:
    _endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT") or os.environ.get("AZURE_OPENAI_API_BASE")
    _api_key = os.environ.get("AZURE_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY")
    _DEPLOYMENT = os.environ.get("AZURE_AI_DEPLOYMENT") or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    if _endpoint and _api_key:
        _CLIENT = AzureOpenAI(
            azure_endpoint=_endpoint,
            api_key=_api_key,
            api_version=os.environ.get("AZURE_API_VERSION", "2024-12-01-preview"),
        )


class EngagementAgent:
    """Agent that provides learner engagement recommendations."""

    async def handle(self, request: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        context = context or {}
        if _CLIENT is None:
            return {
                "agent": "engagement",
                "response": "Azure client not configured; default advice: study during low-meeting windows such as early mornings or late afternoons and avoid heavy calendar blocks.",
            }

        system_prompt = (
            "You are EngagementAgent. Given a learner's workload signals such as meeting hours, "
            "busy days, and available study presence, recommend the best times to study. "
            "Do not offer generic reminders; instead, suggest precise windows, explain why they "
            "work, and account for productivity patterns."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request},
        ]

        try:
            if hasattr(_CLIENT.chat.completions, "create"):
                resp = await asyncio.to_thread(
                    _CLIENT.chat.completions.create,
                    model=_DEPLOYMENT,
                    messages=messages,
                    temperature=0.2,
                    max_completion_tokens=700,
                )
            else:
                resp = await _CLIENT.chat.completions.create(
                    model=_DEPLOYMENT,
                    messages=messages,
                    temperature=0.2,
                    max_completion_tokens=700,
                )

            if hasattr(resp, "choices") and resp.choices:
                choice = resp.choices[0]
                if hasattr(choice, "message"):
                    text = choice.message.content
                else:
                    text = choice.get("text") if isinstance(choice, dict) else None
            else:
                text = str(resp)

            return {"agent": "engagement", "response": text}
        except Exception as exc:
            return {"error": "request failed", "detail": str(exc)}
