"""Demo runner for the certification agent system."""
from __future__ import annotations

import argparse
import asyncio

from agents.learning_path_agent import LearningPathAgent
from agents.study_plan_agent import StudyPlanAgent
from agents.assessment_agent import AssessmentAgent


async def run_demo(user_input: str) -> None:
    # Instantiate agents directly
    learning_path_agent = LearningPathAgent()
    study_plan_agent = StudyPlanAgent()
    assessment_agent = AssessmentAgent()

    # Step 1: Learning Path
    print("\n=== Learning Path Agent ===")
    learning_path_result = await learning_path_agent.handle(user_input)
    response_text = learning_path_result.get("response", learning_path_result.get("error", str(learning_path_result)))
    print(response_text)

    # Step 2: Study Plan
    print("\n=== Study Plan Agent ===")
    study_plan_request = (
        f"Based on the target certification and available study hours, build a weekly study schedule. "
        f"Use this request as context: {user_input}"
    )
    study_plan_result = await study_plan_agent.handle(study_plan_request)
    response_text = study_plan_result.get("response", study_plan_result.get("error", str(study_plan_result)))
    print(response_text)

    # Step 3: Assessment
    print("\n=== Assessment Agent ===")
    assessment_request = (
        f"Generate practice questions for the certification topic in this request: {user_input}. "
        "Include source references like exam modules or domains."
    )
    assessment_result = await assessment_agent.handle(assessment_request)
    response_text = assessment_result.get("response", assessment_result.get("error", str(assessment_result)))
    print(response_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the certification agent demo flow.")
    parser.add_argument("request", nargs="+", help="User request for the demo flow")
    args = parser.parse_args()

    user_input = " ".join(args.request)
    asyncio.run(run_demo(user_input))


if __name__ == "__main__":
    main()
