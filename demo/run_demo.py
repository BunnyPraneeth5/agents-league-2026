"""Demo runner for the certification agent system."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.learning_path_agent import LearningPathAgent
from agents.study_plan_agent import StudyPlanAgent
from agents.assessment_agent import AssessmentAgent


def print_agent_response(title: str, result: dict) -> None:
    print(f"\n=== {title} ===")
    print(result.get("response") or result.get("detail") or result.get("error", "No response returned."))


async def run_demo(user_input: str) -> None:
    # Step 1: Learning Path
    learning_path_agent = LearningPathAgent()
    learning_path_result = await learning_path_agent.handle(user_input, {"original_request": user_input})
    learning_path_response = learning_path_result.get("response", "")
    print_agent_response("Learning Path Agent", learning_path_result)

    # Step 2: Study Plan
    study_plan_agent = StudyPlanAgent()
    study_plan_context = {
        "original_request": user_input,
        "learning_path": learning_path_response,
    }
    study_plan_result = await study_plan_agent.handle(user_input, study_plan_context)
    study_plan_response = study_plan_result.get("response", "")
    print_agent_response("Study Plan Agent", study_plan_result)

    # Step 3: Assessment
    assessment_agent = AssessmentAgent()
    assessment_context = {
        "original_request": user_input,
        "study_plan": study_plan_response,
    }
    assessment_result = await assessment_agent.handle(user_input, assessment_context)
    print_agent_response("Assessment Agent", assessment_result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the certification agent demo flow.")
    parser.add_argument("request", nargs="+", help="User request for the demo flow")
    args = parser.parse_args()

    user_input = " ".join(args.request)
    asyncio.run(run_demo(user_input))


if __name__ == "__main__":
    main()
