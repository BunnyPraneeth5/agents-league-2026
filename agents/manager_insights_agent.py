"""Manager insights agent: produces manager-facing team health summaries."""
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


class ManagerInsightsAgent:
    """Agent that produces manager-facing insights and reports."""

    async def handle(self, request: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        context = context or {}
        if _CLIENT is None:
            return {
                "agent": "manager_insights",
                "response": "Azure client not configured; default summary: team progress is mixed, identify learners without study plans and flag those behind schedule.",
            }

        system_prompt = (
            "You are ManagerInsightsAgent. Summarize how a certification team is performing overall. "
            "Identify who is behind schedule, who is at risk of failing, and any opportunities to improve "
            "the team's progress. Produce a concise manager-facing summary."
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
                    max_completion_tokens=800,
                )
            else:
                resp = await _CLIENT.chat.completions.create(
                    model=_DEPLOYMENT,
                    messages=messages,
                    temperature=0.2,
                    max_completion_tokens=800,
                )

            if hasattr(resp, "choices") and resp.choices:
                choice = resp.choices[0]
                if hasattr(choice, "message"):
                    text = choice.message.content
                else:
                    text = choice.get("text") if isinstance(choice, dict) else None
            else:
                text = str(resp)

            return {"agent": "manager_insights", "response": text}
        except Exception as exc:
            return {"error": "request failed", "detail": str(exc)}
