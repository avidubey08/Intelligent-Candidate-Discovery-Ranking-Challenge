---
title: Redrob Candidate Ranker
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "5.0.0"
python_version: "3.10"
app_file: app.py
pinned: true
short_description: Ranks 100K ML/AI candidates in 37 seconds — no LLM, no GPU
---

# Redrob Candidate Ranker

> **India Runs Data & AI Challenge** — Hackathon Submission

Discovers and ranks the **top 100 ML/AI engineering candidates** from 100,000 profiles in under **37 seconds on CPU**.

## How to use

1. Upload your `candidates.jsonl` file (or click **Run Sample Demo** to try instantly)
2. Click **Run Full Pipeline**
3. See the ranked top-10 in the results table
4. Download `submission.csv` with all 100 ranked candidates

## Pipeline stages

| Stage | What happens |
|-------|-------------|
| A + B | Ingest candidates → extract 35 features across 5 JD-aligned buckets |
| C | Transparent weighted scoring (no opaque model) |
| D | Two-layer honeypot detection: rules + Isolation Forest |
| E | O(N log K) heap selects top-100 |
| F + G | Template-driven reasoning text → submission.csv |

## Reproduce locally

```bash
git clone <your-repo>
pip install -r requirements.txt
python rank.py --candidates India_runs_data_and_ai_challenge/candidates.jsonl --out submission.csv
```
