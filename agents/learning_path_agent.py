"""Learning path agent: recommends certification learning paths.

This agent uses the Azure OpenAI `AzureOpenAI` client. It builds a
specialized system prompt, sends the request to the configured
Azure endpoint, and returns the model's textual response.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from tools.search_knowledge import search_knowledge

load_dotenv()

try:
    from openai import AzureOpenAI
except Exception:  # pragma: no cover - runtime dependency
    AzureOpenAI = None

# Initialize client eagerly so errors surface early
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


class LearningPathAgent:
    """Agent that generates or recommends certification learning paths."""

    async def handle(self, request: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        context = context or {}
        if _CLIENT is None:
            return {
                "agent": "learning_path",
                "response": "Azure client not configured; default recommendation: Start with core exam objectives, divide study into fundamentals, hands-on labs, and practice tests.",
            }

        system_prompt = (
            "You are LearningPathAgent, an expert in designing certification learning "
            "paths. Given a user's request, produce a concise, actionable learning path "
            "with recommended resources, approximate durations, and suggested practice."
        )
        knowledge_results = await search_knowledge(request)
        if knowledge_results:
            grounded_context = "\n\n".join(
                f"Source: {result['source']}\nScore: {result['score']}\nContent: {result['content']}"
                for result in knowledge_results
                if result.get("content")
            )
            if grounded_context:
                system_prompt = (
                    f"{system_prompt}\n\n"
                    f"Use this grounded knowledge to inform your response:\n{grounded_context}"
                )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request},
        ]

        # Use threadpool to avoid blocking event loop if client is synchronous
        try:
            if hasattr(_CLIENT.chat.completions, "create"):
                # sync create -> run in thread
                resp = await asyncio.to_thread(
                    _CLIENT.chat.completions.create,
                    model=_DEPLOYMENT,
                    messages=messages,
                    temperature=0.2,
                    max_completion_tokens=800,
                )
            else:
                # attempt async call
                resp = await _CLIENT.chat.completions.create(
                    model=_DEPLOYMENT,
                    messages=messages,
                    temperature=0.2,
                    max_completion_tokens=800,
                )

            # extract text from common response shapes
            text = None
            if hasattr(resp, "choices") and resp.choices:
                choice = resp.choices[0]
                if hasattr(choice, "message"):
                    text = choice.message.content
                else:
                    text = choice.get("text") if isinstance(choice, dict) else None
            else:
                # fallback to string representation
                text = str(resp)

            return {"agent": "learning_path", "response": text}
        except Exception as exc:
            return {"error": "request failed", "detail": str(exc)}

