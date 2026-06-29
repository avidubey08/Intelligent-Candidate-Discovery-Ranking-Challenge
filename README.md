# 🤖 Redrob — Intelligent Candidate Discovery & Ranking

> **India Runs Data & AI Challenge** · Hackathon Submission
> *Intelligent Candidate Discovery & Ranking Challenge*

---

## 🏆 Project Overview

This project is a complete **batch candidate ranking pipeline** built for the Redrob Hackathon Challenge. Given a dataset of **100,000 candidate profiles**, the system discovers and ranks the **top 100 most suitable candidates** for a Senior ML/AI Engineering role — in under **37 seconds on a CPU-only machine**.

The pipeline is fully transparent, inspectable, and defensible at every stage. Every score decomposes into named feature contributions. Every reasoning sentence is generated directly from the same feature values that drove the rank — no hallucination possible, no black-box model.

---

## ⚡ Performance at a Glance

| Metric | Result |
|--------|--------|
| Dataset size | **100,000 candidates** |
| Total wall-clock time | **37.3 seconds** (limit: 5 min) |
| Feature extraction (100K records) | 34.4s |
| Isolation Forest training | 1.1s |
| Batch anomaly scoring (vectorized) | 1.0s |
| Scoring + Top-100 heap selection | 0.8s |
| Official validation result | ✅ **Submission is valid** |
| Top-1 candidate score | 0.6364 |
| Top-10 minimum score | 0.5675 |
| Top-50 minimum score | 0.5195 |
| Honeypots in top-100 | **0** (all tenure ratios clean) |

---

## 🎯 The Core Constraints

> *No LLM calls. No GPU. No network. No hardcoded honeypot lists. Must run in 5 minutes on 16 GB RAM, CPU-only.*

These constraints shaped every architectural decision:

| Constraint | Our Design Response |
|-----------|---------------------|
| 5-min CPU limit | Stream-parse JSONL line-by-line; O(N log K) heap selection; vectorized batch ISO scoring |
| 80% score = NDCG@10 + NDCG@50 | Optimize top-of-list precision; multiplicative availability (not additive) |
| Stage 4: manual reasoning audit | Template-driven reasoning from real feature values only — no LLM |
| Stage 5: 30-min defend interview | Every weight documented with a JD-sourced rationale |
| Honeypot rate >10% = disqualified | Two-layer detection: rules + Isolation Forest; never a hardcoded denylist |

---

## 🏗️ Architecture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                          rank.py  (Entrypoint)                          │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
         ┌───────────────▼──────────────┐
         │  Stage A: Ingest & Validate  │  candidates.jsonl / .jsonl.gz
         │  Stream-parse line-by-line   │  Auto-detects JSON array format too
         └───────────────┬──────────────┘
                         │ 100K candidate dicts (generator)
         ┌───────────────▼──────────────┐
         │  Stage B: Feature Extraction │  features.py
         │  ~35 features, 5 JD-buckets  │  Pure flat dict output, unit-testable
         └───────────────┬──────────────┘
                         │ all_features list (RAM: ~800 MB for 100K)
         ┌───────────────▼──────────────┐
         │  Stage D: Consistency Audit  │  consistency_audit.py
         │  Rule checks + Isolation     │  Batch vectorized over full 100K
         │  Forest — batch scored once  │  One numpy matrix call — not row-by-row
         └───────────────┬──────────────┘
                         │ iso_scores list
         ┌───────────────▼──────────────┐
         │  Stage C: Scoring            │  scoring.py
         │  Transparent weighted sum    │  base_fit x availability x (1-suspicion)
         └───────────────┬──────────────┘
                         │ scored stream (generator)
         ┌───────────────▼──────────────┐
         │  Stage E: Select Top-100     │  select_top_k.py
         │  O(N log K) bounded min-heap │  Tie-break embedded in heap key
         └───────────────┬──────────────┘
                         │ top-100 items (sorted)
         ┌───────────────▼──────────────┐
         │  Stage F: Reasoning          │  reasoning.py
         │  Template-driven text        │  From same features that drove score
         │  No LLM — zero hallucination │
         └───────────────┬──────────────┘
                         │
         ┌───────────────▼──────────────┐
         │  Stage G: Emit & Validate    │  submission.csv (UTF-8)
         │  candidate_id,rank,score,    │  validate_submission.py → valid ✅
         │  reasoning                   │
         └──────────────────────────────┘
```

---

## 📐 Feature Engineering in Depth (Stage B)

The `features.py` module extracts **35+ numeric features** grouped into 5 JD-aligned buckets. All term dictionaries are **fixed and documented** — no ad-hoc or dynamic matching.

---

### Bucket 1 — Skill / Role Fit (40% of base score)

The decisive **anti-keyword-stuffing** signal. A candidate in an actual ML/AI title with modest skills outranks a keyword-stuffed profile with the wrong job function.

| Feature | Source Field | Computation |
|---------|-------------|-------------|
| `title_ml_score` | `profile.current_title` | Exact substring match vs 24 ML/AI function keywords → 0, 0.35, or 1.0 |
| `core_skill_score` | `skills[].name/proficiency/endorsements/duration_months` | `proficiency_weight × (0.4 + 0.35×log1p(end)/log1p(100) + 0.25×log1p(dur)/log1p(60))` |
| `core_skill_count` | `skills[]` | Count of skills matching a 50-term JD-aligned vocabulary |
| `github_activity` | `redrob_signals.github_activity_score` | Normalized 0–1 (−1 if no GitHub) |
| `years_experience` | `profile.years_of_experience` | Log-scaled, diminishing returns after 10 yrs |

**Why log-scale endorsements × duration?**
An `expert`-proficiency skill with 0 endorsements and 2 months duration is the exact keyword-stuffing pattern the JD warns about. `log1p(0) = 0` → collapses that skill's contribution to near-zero automatically.

---

### Bucket 2 — Career Substance (35% of base score)

Detects **shipped production systems** using a fixed 30-term vocabulary mined directly from the JD's own language.

**Production Vocabulary (documented proxy-term dictionary):**

```text
shipped, production, deployed, launched, live, real users,
docker, kubernetes, ci/cd, latency, throughput, on-call, incident,
sla, a/b test, experimentation, recommendation, ranking, retrieval,
model serving, real-time, scaling, monitoring, rollback, pipeline
```

| Feature | Logic |
|---------|-------|
| `production_score` | `min(1, log1p(hits) / log1p(30))` — log-scaled count, cap at 30 hits |
| `consulting_only_flag` | All roles at TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini → 0.25 penalty |
| `research_only_flag` | Academic-only history, zero production language → hard floor score (0.02) |
| `title_chasing_penalty` | 3+ company hops in ≤18-month stints → up to 0.50 penalty |
| `career_company_product_ratio` | Fraction of roles at non-consulting product companies |

---

### Bucket 3 — Logistics Fit (15% of base score)

| Feature | Logic |
|---------|-------|
| `location_score` | India + Tier-1 city → 1.0; India + willing to relocate → 0.85; India only → 0.65; outside India + relocate → 0.45; outside + won't relocate → 0.20 |
| `notice_score` | **Plateau** ≤30 days → 1.0 (JD buyout caveat); rising linear penalty past 30 days; floor at 0.10 |
| `work_mode_score` | flexible→1.0, hybrid→0.95, onsite→0.85, remote→0.70 |

**Why plateau shape for notice period?**
The JD explicitly states: *"they can absorb a 30-day buyout."* A cliff-edge penalty at 31 days would penalize candidates the JD itself deems acceptable. The plateau-then-slope shape encodes this policy precisely.

---

### Bucket 4 — Behavioral Availability (Multiplicative Factor)

> *"A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is, for hiring purposes, not actually available. Down-weight them appropriately."*
> — Job Description, Availability Section

Applied **multiplicatively** on base fit — not additively. This prevents a strong skill score from swamping a clearly-unavailable candidate.

```python
availability_score = (
    0.35 * recency_factor            # exp(-lambda * days_inactive), half-life = 30 days
  + 0.25 * recruiter_response_rate   # direct [0,1] signal
  + 0.20 * (1.0 if open_to_work else 0.5)
  + 0.12 * interview_completion_rate
  + 0.08 * response_time_factor      # log-inverse of avg_response_time_hours
)
```

**Non-linear decay design rationale:**

| Signal | Decay Shape | Rationale |
|--------|-------------|-----------|
| `last_active_date` | Exponential, half-life 30 days | Recency matters far more in first weeks than after months of silence |
| `avg_response_time_hours` | Log-inverse (capped at 336h) | 2h vs 8h is noise; 8h vs 72h is genuine signal |
| `notice_period_days` | Plateau then linear slope | Matches JD's explicit 30-day buyout policy |

---

### Bucket 5 — Consistency / Audit Features (Fed to Stage D)

Not used directly in fit scoring. These 6 features feed the Isolation Forest as its input matrix.

| Feature | Honeypot Signature |
|---------|-------------------|
| `tenure_yoe_ratio` | >1.35× → claimed months significantly exceed stated YOE |
| `impossible_tenure_flag` | Single role duration longer than total stated YOE |
| `expert_short_skill_count` | Expert-proficiency skill with <12 months practice |
| `proficiency_duration_mismatch_rate` | Fraction of skills with contradictory proficiency vs duration |
| `n_skills` | Unusually high skill count (used as context feature) |
| `n_roles` | Number of career history entries |

---

## 🕵️ Honeypot Detection — Two-Layer System (Stage D)

### Layer 1: Rule-Based Checks (Explainable)

Each rule produces a one-sentence flag for interview defense:

- *"Claimed career months are 2.1× stated YOE"*
- *"3 expert-level skills claimed with under 12 months practice"*
- *"Proficiency-duration mismatch rate: 64% of skills"*

Rules contribute 65% weight to the combined suspicion score.

### Layer 2: Isolation Forest (General Anomaly Detection)

- Trained on **6 engineered consistency features** (not raw JSON fields) — so contradictions are already encoded as numeric anomalies
- 100 estimators, `contamination=0.05`, `random_state=42`, `n_jobs=-1`
- **Batch-scored in one vectorized numpy call** across all 100K:

```python
raw_scores = iso_model.decision_function(X)   # X shape: (100000, 6)
```

This avoids 100K individual sklearn calls (which caused a 60+ second stall in testing).

```python
combined_suspicion = 0.65 × rule_score + 0.35 × iso_score
final_score        = base_fit × availability × (1 − 0.40 × combined_suspicion)
```

**Why both layers?**

- Rules → explainability for Stage 5 interview ("*here is the specific contradiction*")
- Isolation Forest → defensible general approach ("*I trained an anomaly detector on consistency features — honeypots surface as outliers without needing to special-case them*")

**Why down-weight, not exclude?**
The spec says: *"a hardcoded honeypot-ID denylist looks like overfitting to known traps and won't generalize to the hidden ground truth's specific 80."* Our pipeline penalizes suspicious profiles by up to 40% — they naturally fall below genuine candidates.

---

## 📊 Scoring Formula

```python
base_fit = 0.40 × skill_fit_score
         + 0.35 × career_substance_score
         + 0.15 × logistics_score
         + 0.10 × education_score

final_score = base_fit × availability_factor × (1 − 0.40 × suspicion_score)
```

All weights live in `scoring.py` with documented rationale. The formula is a **transparent weighted sum** — no opaque model, every contribution is named and retrievable per candidate.

**Weight rationale:**

- **Skill fit (40%)** — title function match is the primary signal; skills are weighted not counted
- **Career substance (35%)** — production evidence is the core differentiator per the JD
- **Logistics (15%)** — location + availability logistics are necessary conditions, not differentiators
- **Education (10%)** — minor signal; tier + STEM field

---

## 🔢 Top-K Selection — O(N log K) Heap (Stage E)

Standard sort of 100K floats is fast, but materializing a full sorted list wastes memory. The bounded heap approach is the architecturally correct pattern:

```python
# heap key: (score, -cid_integer)
# min-heap root = worst item (lowest score; on tie: highest numeric cid = evictee)
# This tie-break exactly matches validate_submission.py lines 136-144

if key > heap[0][0]:
    heapq.heapreplace(heap, (key, item))

# Final sort of just 100 items: trivial cost
results.sort(key=lambda x: (-x["score"], cid_int(x["candidate_id"])))
```

---

## 💬 Reasoning Generation (Stage F)

No LLM. Template-driven from the same feature values that produced the score.

**Example output:**

```text
Senior AI Engineer with 5.9 yrs exp; 13 core ML skills; in ML/AI function
(at Apple; strong production evidence). Availability: open to work; active
this week; response rate 82%; notice 30d.
```

**Why this passes all 6 Stage-4 manual review checks:**

| Check | How we pass |
|-------|------------|
| Specific facts | All values from actual candidate fields |
| JD connection | Bucket names map to JD language exactly |
| Honest concerns | Low-scoring buckets surface as caveats automatically |
| No hallucination | Text only references fields we actually read |
| Variation | Different top contributors → different sentences across candidates |
| Rank consistency | Text IS the score's explanation — mathematically cannot contradict rank |

---

## 📁 Project Structure

```text
Redrob/
├── rank.py                    # Entrypoint — orchestrates all 7 stages
├── features.py                # Stage B — 35+ features, 5 JD-aligned buckets
├── scoring.py                 # Stage C — all weights live here, nowhere else
├── consistency_audit.py       # Stage D — rules + Isolation Forest
├── select_top_k.py            # Stage E — O(N log K) bounded heap
├── reasoning.py               # Stage F — template-driven text, no LLM
├── requirements.txt           # Pinned dependencies
├── submission_metadata.yaml   # Team info, AI tools declaration, methodology
├── README.md                  # This file
├── submission.csv             # Final output — validated ✅
├── tests/
│   └── test_features.py       # 6 unit tests — all passing ✅
└── India_runs_data_and_ai_challenge/
    ├── candidates.jsonl       # 100K candidate profiles (~487 MB)
    ├── sample_candidates.json # 50-candidate sample for development
    ├── candidate_schema.json  # JSON schema
    ├── validate_submission.py # Official validator — run before submitting
    └── submission_metadata_template.yaml
```

---

## 🚀 Setup & Reproduce

### Requirements

- Python 3.10+
- 16 GB RAM
- CPU-only (no GPU required or used)
- No network access during ranking

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Reproduce (single command)

```bash
python rank.py --candidates India_runs_data_and_ai_challenge\candidates.jsonl --out submission.csv
```

Works with gzipped input too:

```bash
python rank.py --candidates India_runs_data_and_ai_challenge\candidates.jsonl.gz --out submission.csv
```

### 3. Validate output

```bash
python India_runs_data_and_ai_challenge/validate_submission.py submission.csv
# Submission is valid.
```

### 4. Run tests

```bash
python tests/test_features.py
# 6/6 tests passed
```

---

## 🧪 Test Coverage

| Test | What it verifies |
|------|-----------------|
| `test_extract_features_keys` | All required feature keys present in output dict |
| `test_scores_in_range` | All bounded features in [0, 1] for first 10 candidates |
| `test_notice_period_plateau` | ≤30 days → 1.0; 90 days → <0.65 |
| `test_availability_decay` | 6-month-inactive candidate → recency_factor < 0.01 |
| `test_consulting_only_flag` | All-consulting career → flag True |
| `test_full_pipeline_sample` | End-to-end extract → score → top-K on 50 sample candidates |

---

## 🏅 Top-10 Ranked Candidates

| Rank | Candidate ID | Score | Title | Company | Location |
|------|-------------|-------|-------|---------|----------|
| 1 | CAND_0002025 | 0.6364 | Senior AI Engineer | Apple | India |
| 2 | CAND_0064326 | 0.6281 | Search Engineer | Sarvam AI | Gurgaon, India |
| 3 | CAND_0077337 | 0.6021 | Staff ML Engineer | Paytm | India |
| 4 | CAND_0039754 | 0.6001 | Senior Applied Scientist | Meta | India |
| 5 | CAND_0046525 | 0.5829 | Senior ML Engineer | Genpact AI | Pune, India |
| 6 | CAND_0046132 | 0.5825 | AI Research Engineer | Verloop.io | Noida, India |
| 7 | CAND_0088025 | 0.5735 | Staff ML Engineer | — | India |
| 8 | CAND_0043637 | 0.5732 | Junior ML Engineer | Rephrase.ai | Delhi, India |
| 9 | CAND_0008295 | 0.5694 | AI Research Engineer | Razorpay | Pune, India |
| 10 | CAND_0007596 | 0.5675 | Junior ML Engineer | Flipkart | Delhi, India |

**All top-10:** India-based ✅ &nbsp;·&nbsp; ML/AI function titles ✅ &nbsp;·&nbsp; Active May 2026 ✅ &nbsp;·&nbsp; Clean tenure ratios (0.98–1.00) ✅

---

## 🚫 What We Deliberately Did NOT Build

| Not built | Reason |
|-----------|--------|
| Per-candidate LLM calls | Banned by spec; 100K × 200ms = 5.5 hrs |
| Hardcoded honeypot ID denylist | Overfits to known traps; won't generalize to hidden ground truth |
| Single opaque ML model | Cannot generate honest reasoning text; cannot defend weights in Stage 5 |
| Live API / microservice | This is a batch script, not a deployed system |
| Custom React/Node frontend | Nothing in rubric scores UI; scope creep only |

---

## 🤖 AI Tools Declaration

Used **Antigravity (Google DeepMind AI assistant)** for architecture discussion, code scaffolding, and review. All ranking logic, feature weights, proxy-term dictionaries, decay curve shapes, and design rationales are hand-authored and documented. No candidate data was fed to any external LLM during pipeline execution.