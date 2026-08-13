# Agentic AI — Learning in Public

A 3-week, build-every-day journey from senior software engineer into applied AI / LLM engineering. Each day ships a small working piece and a short writeup of what I built, what I learned, and what broke along the way.

**Background:** 13 years shipping production software (mobile + cloud), AWS-certified. This log is me going deep on building and deploying LLM systems — RAG, agents, evals, AWS-native deployment — with production discipline, not just prototypes.

## Why "learning in public"
Every day has an honest writeup, including the errors and how I worked through them. The debugging stories are the point: they're the real evidence of how I think, not a tutorial rehash.

## The log

| Day | Focus | Writeup |
|----|-------|---------|
| 1 | Streaming FastAPI endpoint over the OpenAI API | [Day 1](ThreeWeeksProgram/Day1/README.md) |
| 2 | Structured JSON output (Pydantic) | _coming up_ |
| 3 | Document ingestion + chunking | _coming up_ |
| ... | ... | ... |

*(Update this table as you go — it becomes the map of the whole journey.)*

## Running any day's code
```bash
cd ThreeWeeksProgram/DayN
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# see that day's README for run + test commands
```

## The arc
Weeks 1–3 build toward a deployed, evaluated RAG system on AWS Bedrock, plus an agentic workflow — the portfolio core for landing an applied AI engineering role.