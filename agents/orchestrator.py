"""Orchestrator for routing user requests to specialized agents.

Features:
- Loads configuration from a .env file
- Configures the OpenAI Python SDK for Azure endpoints
- Routes requests to one of: LearningPathAgent, StudyPlanAgent,
  EngagementAgent, AssessmentAgent, ManagerInsightsAgent
- Optional model-based intent classification using Azure OpenAI
- Uses asyncio for concurrency and timeouts

The orchestrator assumes each agent exposes an async `handle(request, context)`
method that returns a serializable result (str or dict).
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from dotenv import load_dotenv

try:
    import openai
except Exception:  # pragma: no cover - fail early if missing in runtime
    openai = None

from agents.learning_path_agent import LearningPathAgent
from agents.study_plan_agent import StudyPlanAgent
from agents.engagement_agent import EngagementAgent
from agents.assessment_agent import AssessmentAgent
from agents.manager_insights_agent import ManagerInsightsAgent

load_dotenv()  # read .env from project root

LOG = logging.getLogger("orchestrator")
logging.basicConfig(level=logging.INFO)


@dataclass
class OrchestratorConfig:
    azure_api_base: str
    azure_api_key: str
    azure_api_version: Optional[str]
    deployment: Optional[str]
    use_model_routing: bool = False
    model_timeout_seconds: int = 8


def _load_config_from_env() -> OrchestratorConfig:
    return OrchestratorConfig(
        azure_api_base=os.getenv("AZURE_AI_PROJECT_ENDPOINT", ""),
        azure_api_key=os.getenv("AZURE_API_KEY", ""),
        azure_api_version=os.getenv("AZURE_API_VERSION", "2024-12-01-preview"),
        deployment=os.getenv("AZURE_AI_DEPLOYMENT", "gpt-5.4-mini"),
        use_model_routing=False,
        model_timeout_seconds=8,
    )


class Orchestrator:
    """Main orchestrator responsible for routing requests to agents."""

    def __init__(self, config: Optional[OrchestratorConfig] = None, agent_timeout: int = 30):
        self.config = config or _load_config_from_env()
        self.agent_timeout = agent_timeout

        if openai is None:
            LOG.warning("`openai` package not available; model routing disabled.")
            self.config.use_model_routing = False
        else:
            self._configure_openai()

        # Instantiate agents. Agents should expose async handle(request, context) -> Any
        self.agents = {
            "learning_path": LearningPathAgent(),
            "study_plan": StudyPlanAgent(),
            "engagement": EngagementAgent(),
            "assessment": AssessmentAgent(),
            "manager_insights": ManagerInsightsAgent(),
        }

        # Simple rule-based keywords to agent mapping as a deterministic fallback
        self.keyword_map = {
            "cert": "learning_path",
            "learn": "learning_path",
            "study": "study_plan",
            "plan": "study_plan",
            "engage": "engagement",
            "motiva": "engagement",
            "assess": "assessment",
            "quiz": "assessment",
            "manager": "manager_insights",
            "insight": "manager_insights",
            "report": "manager_insights",
        }

    def _configure_openai(self) -> None:
        # Configure the OpenAI SDK for Azure endpoints
        try:
            openai.api_type = os.getenv("OPENAI_API_TYPE", "azure")
            openai.api_base = self.config.azure_api_base
            if self.config.azure_api_version:
                openai.api_version = self.config.azure_api_version
            openai.api_key = self.config.azure_api_key
            LOG.info("Configured OpenAI SDK for Azure endpoint: %s", self.config.azure_api_base)
        except Exception as exc:  # pragma: no cover - defensive
            LOG.exception("Failed to configure OpenAI SDK: %s", exc)
            self.config.use_model_routing = False

    async def _classify_with_model(self, user_request: str) -> Optional[str]:
        """Ask the configured Azure OpenAI deployment to select an agent.

        Returns the canonical agent key on success, otherwise None.
        """
        if not self.config.deployment:
            LOG.debug("No deployment configured for model routing.")
            return None

        system = (
            "You are an intent classifier. Choose the most appropriate agent from: "
            "learning_path, study_plan, engagement, assessment, manager_insights. "
            "Return only the agent keyword as a single word in the response."
        )
        prompt = (
            f"User request:\n" f"{user_request}\n\n" f"Which agent should handle this?"
        )

        try:
            # Use ChatCompletion; be tolerant of SDK differences in azure param names
            create_kwargs = {
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                "temperature": 0.0,
            }

            # new versions use `deployment_id` or `engine` as kw; attempt common keys
            if "deployment_id" in openai.ChatCompletion.acreate.__code__.co_varnames:
                create_kwargs["deployment_id"] = self.config.deployment
            else:
                create_kwargs["engine"] = self.config.deployment

            coro = openai.ChatCompletion.acreate(**create_kwargs)
            resp = await asyncio.wait_for(coro, timeout=self.config.model_timeout_seconds)

            # extract text from response (support different shapes)
            text = None
            if hasattr(resp, "choices") and resp.choices:
                # typical shape
                choice = resp.choices[0]
                text = getattr(choice, "message", {}).get("content") if hasattr(choice, "message") else choice.get("text")
            else:
                # fallback: try str()
                text = str(resp)

            if not text:
                return None

            # sanitize
            agent_candidate = text.strip().lower().split()[0]
            if agent_candidate in self.agents:
                LOG.debug("Model classified to agent: %s", agent_candidate)
                return agent_candidate
            LOG.debug("Model returned unknown agent: %s", agent_candidate)
            return None

        except Exception as exc:  # pragma: no cover - runtime integration
            LOG.exception("Model routing failed: %s", exc)
            return None

    def _rule_based_classify(self, user_request: str) -> str:
        """Simple deterministic classifier based on keywords."""
        text = (user_request or "").lower()
        for kw, agent in self.keyword_map.items():
            if kw in text:
                LOG.debug("Rule-based matched keyword '%s' -> %s", kw, agent)
                return agent
        # default fallback
        return "learning_path"

    async def route(self, user_request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Route the user request to the selected agent and return its response.

        Returns a dict containing: agent, result, metadata
        """
        if not user_request or not user_request.strip():
            raise ValueError("user_request must be a non-empty string")

        context = context or {}

        # Decide agent: model-based routing if enabled and available
        agent_key = None
        if self.config.use_model_routing:
            agent_key = await self._classify_with_model(user_request)

        if not agent_key:
            agent_key = self._rule_based_classify(user_request)

        agent = self.agents.get(agent_key)
        if agent is None:
            LOG.error("No agent instance found for key: %s", agent_key)
            raise RuntimeError(f"No agent available for key: {agent_key}")

        try:
            # Support for concurrency: if agents return coroutines, await them with timeout
            coro = agent.handle(user_request, context)
            result = await asyncio.wait_for(coro, timeout=self.agent_timeout)
            return {"agent": agent_key, "result": result, "metadata": {}}
        except asyncio.TimeoutError:
            LOG.exception("Agent %s timed out", agent_key)
            raise
        except Exception as exc:
            LOG.exception("Agent %s raised an exception: %s", agent_key, exc)
            raise


def cli_main() -> None:
    """Small CLI for quick manual testing."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run the certification orchestrator")
    parser.add_argument("request", nargs="+", help="User request text")
    parser.add_argument("--timeout", type=int, default=30, help="Per-agent timeout seconds")
    args = parser.parse_args()

    text = " ".join(args.request)
    orchestrator = Orchestrator(agent_timeout=args.timeout)

    async def _run():
        res = await orchestrator.route(text)
        print(json.dumps(res, indent=2))

    asyncio.run(_run())


if __name__ == "__main__":
    cli_main()
