# CareerForge AI

An Agents League 2026 hackathon project that helps engineering teams plan, schedule, assess, and manage Microsoft certification readiness using a chained multi-agent workflow.

## Architecture Overview

Certification Agent League is built around five specialized Python agents. The demo flow uses the first three agents as a pipeline, while the remaining two agents support workload-aware engagement and manager reporting.

The primary chain is:

1. **LearningPathAgent** receives the learner's original certification goal and creates a certification learning path.
2. **StudyPlanAgent** receives the original request plus the learning path output and turns it into a practical study schedule.
3. **AssessmentAgent** receives the original request plus the study plan output and generates practice questions for the first study topics.

Supporting agents:

4. **EngagementAgent** uses workload signals to recommend better learning windows.
5. **ManagerInsightsAgent** summarizes team-level certification progress and risk areas.

The system also includes an `Orchestrator` for general request routing, but the demo pipeline calls the agents directly so the chained behavior is easy to inspect.

## Microsoft IQ Layer

This project uses a Microsoft Foundry IQ-style knowledge layer backed by Azure AI Search.

The knowledge base is represented by three synthetic documents in the `knowledge/` folder:

- `knowledge/cert_guide.md`: internal certification guide for AZ-204, AZ-400, AZ-305, and DP-203
- `knowledge/team_report.md`: synthetic quarterly learning performance summary
- `knowledge/workload_insights.md`: synthetic workload and learning correlation report

The `tools/search_knowledge.py` helper connects to Azure AI Search using:

```text
AZURE_SEARCH_ENDPOINT
AZURE_SEARCH_KEY
AZURE_SEARCH_INDEX_NAME
```

`LearningPathAgent` queries this knowledge base before calling Azure OpenAI. When search results are available, retrieved snippets are added to the model prompt as grounded context. If the search service is unavailable or returns no results, the agent falls back to its normal prompt behavior.

## How To Run

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Configure Azure OpenAI settings in `.env`:

```bash
AZURE_AI_PROJECT_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_API_KEY=<your-api-key>
AZURE_API_VERSION=2024-12-01-preview
AZURE_AI_DEPLOYMENT=gpt-5.4-mini
```

3. Configure Azure AI Search for the Foundry IQ knowledge layer:

```bash
AZURE_SEARCH_ENDPOINT=https://<your-search-service>.search.windows.net
AZURE_SEARCH_KEY=<your-search-key>
AZURE_SEARCH_INDEX_NAME=<your-index-name>
```

4. Run the demo:

```bash
python demo/run_demo.py "I want to prepare for AZ-204 certification in 4 weeks with 6 hours per week"
```

Expected output format:

```text
=== Learning Path Agent ===
<learning path response>

=== Study Plan Agent ===
<study plan response>

=== Assessment Agent ===
<assessment response>
```

## Agent Descriptions

**LearningPathAgent** designs certification learning paths for engineering learners. It uses the learner's request and, when available, grounded snippets from the Foundry IQ knowledge base to recommend focus areas, sequencing, resources, study duration, and practice activities.

**StudyPlanAgent** turns a learning path into a realistic schedule. It receives the original learner request plus the learning path response, then builds a week-by-week or day-by-day plan that accounts for target certification, available hours, topic order, and pacing.

**AssessmentAgent** generates certification practice questions from the study plan. In the chained demo, it receives the original request plus the study plan and creates questions specifically for the first week of learning so assessment is aligned with the learner's immediate study scope.

**EngagementAgent** recommends when and how a learner should study based on workload context. It is intended to use meeting load, focus hours, and preferred learning windows to help learners protect high-quality study time and reduce schedule friction.

**ManagerInsightsAgent** summarizes team certification performance for managers. It highlights progress, pass readiness, at-risk learners, and recommended interventions using synthetic learner and workload data.

## Synthetic Data Disclaimer

This repository includes synthetic data only. The learner names, employee IDs, workload values, study outcomes, and reports are fabricated for demonstration purposes. No real employee data, emails, or personally identifiable information are included.

Synthetic data files:

- `data/synthetic_learners.json`
- `data/synthetic_workload.json`
- `data/certifications.json`

Synthetic knowledge files:

- `knowledge/cert_guide.md`
- `knowledge/team_report.md`
- `knowledge/workload_insights.md`

## Tech Stack

- Python
- Azure OpenAI `gpt-5.4-mini`
- Azure AI Search
- Microsoft Foundry
- Microsoft Foundry IQ knowledge base pattern
