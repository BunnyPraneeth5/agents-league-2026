"""Study plan agent: builds weekly study schedules."""
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


class StudyPlanAgent:
    """Agent that builds study plans for learners."""

    async def handle(self, request: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        context = context or {}
        if _CLIENT is None:
            return {
                "agent": "study_plan",
                "response": "Azure client not configured; default weekly plan: focus on core objectives in week 1, labs and practice in week 2, and review in week 3.",
            }

        system_prompt = (
            "You are StudyPlanAgent. A learner has asked for a weekly study schedule for a "
            "specific certification. Use the target certification and available weekly hours to "
            "produce a day-by-day or week-by-week plan, including focus areas, recommended study "
            "blocks, and realistic pacing. If the certification or hours are missing, ask clearly."
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

            return {"agent": "study_plan", "response": text}
        except Exception as exc:
            return {"error": "request failed", "detail": str(exc)}
