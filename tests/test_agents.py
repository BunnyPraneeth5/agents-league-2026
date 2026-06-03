import asyncio

from agents.orchestrator import Orchestrator, OrchestratorConfig


def make_config() -> OrchestratorConfig:
    return OrchestratorConfig(
        azure_api_base="",
        azure_api_key="",
        azure_api_version=None,
        deployment=None,
        use_model_routing=False,
        model_timeout_seconds=1,
    )


def test_routing_learning_path():
    orch = Orchestrator(config=make_config())
    res = asyncio.run(orch.route("I want to learn for the certification"))
    assert res["agent"] == "learning_path"


def test_routing_study_plan():
    orch = Orchestrator(config=make_config())
    res = asyncio.run(orch.route("Can you create a study plan for me?"))
    assert res["agent"] == "study_plan"


def test_routing_engagement():
    orch = Orchestrator(config=make_config())
    res = asyncio.run(orch.route("How can I increase engagement?") )
    assert res["agent"] == "engagement"


def test_routing_assessment():
    orch = Orchestrator(config=make_config())
    res = asyncio.run(orch.route("Run an assessment quiz"))
    assert res["agent"] == "assessment"


def test_routing_manager_insights():
    orch = Orchestrator(config=make_config())
    res = asyncio.run(orch.route("Generate a manager report"))
    assert res["agent"] == "manager_insights"
