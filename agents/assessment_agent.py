"""Assessment agent: generates certification practice questions."""
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


class AssessmentAgent:
    """Agent that runs assessments and returns results."""

    async def handle(self, request: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        context = context or {}
        if _CLIENT is None:
            return {
                "agent": "assessment",
                "response": "Azure client not configured; default practice prompt: create 3 certification practice questions and identify source modules for each.",
            }

        system_prompt = (
            "You are AssessmentAgent. Generate 3-5 practice questions for the certification topic in the user request. "
            "Each question should include a source note such as 'based on AZ-204 module 3' or a comparable exam domain. "
            "Provide the question, answer choices if appropriate, and indicate the correct answer."
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
                    max_completion_tokens=900,
                )
            else:
                resp = await _CLIENT.chat.completions.create(
                    model=_DEPLOYMENT,
                    messages=messages,
                    temperature=0.2,
                    max_completion_tokens=900,
                )

            if hasattr(resp, "choices") and resp.choices:
                choice = resp.choices[0]
                if hasattr(choice, "message"):
                    text = choice.message.content
                else:
                    text = choice.get("text") if isinstance(choice, dict) else None
            else:
                text = str(resp)

            return {"agent": "assessment", "response": text}
        except Exception as exc:
            return {"error": "request failed", "detail": str(exc)}
