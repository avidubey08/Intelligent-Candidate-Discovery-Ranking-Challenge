"""
app.py — Redrob Candidate Ranker · HuggingFace Spaces Demo
Gradio interface wrapping the full 7-stage ranking pipeline.
"""

import gradio as gr
import json
import csv
import time
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from features import extract_features
from consistency_audit import train_isolation_forest, compute_suspicion
from scoring import compute_score
from select_top_k import select_top_k
from reasoning import generate_reasoning

# ── CSS ─────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif !important; }

body, .gradio-container {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
    min-height: 100vh;
}

.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
}

/* Header */
.header-box {
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(168,85,247,0.10));
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    text-align: center;
    backdrop-filter: blur(10px);
}

/* Stat cards */
.stat-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: all 0.2s;
}
.stat-card:hover {
    background: rgba(99,102,241,0.1);
    border-color: rgba(99,102,241,0.4);
    transform: translateY(-2px);
}

/* Buttons */
.primary-btn {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 12px 28px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important;
}
.primary-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.5) !important;
}

.secondary-btn {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: rgba(255,255,255,0.85) !important;
    font-weight: 500 !important;
    padding: 12px 28px !important;
    transition: all 0.2s !important;
}
.secondary-btn:hover {
    background: rgba(255,255,255,0.10) !important;
    border-color: rgba(255,255,255,0.3) !important;
}

/* Panel */
.panel {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 16px;
}

/* Table */
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}
th {
    background: rgba(99,102,241,0.2);
    color: rgba(255,255,255,0.9);
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    font-size: 11px;
}
td {
    padding: 10px 14px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    color: rgba(255,255,255,0.82);
    vertical-align: top;
}
tr:hover td { background: rgba(99,102,241,0.07); }
tr:last-child td { border-bottom: none; }
.rank-badge {
    display: inline-block;
    background: linear-gradient(135deg,#6366f1,#8b5cf6);
    color: white;
    border-radius: 6px;
    padding: 2px 8px;
    font-weight: 700;
    font-size: 12px;
    min-width: 28px;
    text-align: center;
}
.score-bar {
    display: inline-block;
    height: 6px;
    border-radius: 3px;
    background: linear-gradient(90deg,#6366f1,#8b5cf6);
    margin-right: 6px;
    vertical-align: middle;
}

/* Status */
.status-ok {
    color: #34d399;
    font-weight: 600;
}
.status-warn {
    color: #fbbf24;
}

/* Timing box */
.timing-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin-top: 16px;
}
"""

# ── Pipeline ─────────────────────────────────────────────────────────────────

def iter_candidates_str(content_str):
    """Parse JSON array or JSONL from string."""
    content_str = content_str.strip()
    if content_str.startswith("["):
        return json.loads(content_str)
    results = []
    for line in content_str.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            results.append(json.loads(line))
        except Exception:
            continue
    return results


def run_pipeline(candidates):
    """Run the full 7-stage pipeline on a list of candidate dicts."""
    timings = {}
    t0 = time.time()

    # Stage B: Feature extraction
    all_features = [extract_features(c) for c in candidates]
    timings["feature_extraction"] = round(time.time() - t0, 2)

    # Stage D: Train Isolation Forest
    t1 = time.time()
    if len(all_features) >= 10:
        iso_model = train_isolation_forest(all_features, contamination=0.05)
    else:
        iso_model = None
    timings["consistency_audit"] = round(time.time() - t1, 2)

    # Stage C+E: Score all + select top-100
    t2 = time.time()
    def scored_stream():
        for f in all_features:
            susp = compute_suspicion(f, iso_model)
            sr   = compute_score(f, susp)
            yield {
                "candidate_id": f["_candidate_id"],
                "score": sr["score"],
                "features": f,
                "score_result": sr,
                "suspicion_result": susp,
            }

    top_k = min(100, len(candidates))
    top = select_top_k(scored_stream(), k=top_k)
    timings["scoring_selection"] = round(time.time() - t2, 2)

    # Stage F: Reasoning
    t3 = time.time()
    rows = []
    for rank_i, item in enumerate(top, start=1):
        reasoning = generate_reasoning(
            item["features"], item["score_result"], item["suspicion_result"]
        )
        rows.append({
            "rank": rank_i,
            "candidate_id": item["candidate_id"],
            "score": round(item["score"], 4),
            "reasoning": reasoning,
            "features": item["features"],
            "score_result": item["score_result"],
            "suspicion": item["suspicion_result"],
        })
    timings["reasoning"] = round(time.time() - t3, 2)
    timings["total"] = round(time.time() - t0, 2)

    return rows, timings


# ── Rendering ─────────────────────────────────────────────────────────────────

def _score_bar_html(score, max_score=0.70):
    pct = int(min(100, score / max_score * 100))
    color = "#34d399" if score > 0.55 else "#6366f1" if score > 0.40 else "#fbbf24"
    return f'<span class="score-bar" style="width:{pct}px;background:{color}"></span>'


def build_table_html(rows, show_top=10):
    if not rows:
        return "<p style='color:rgba(255,255,255,0.5);text-align:center;padding:40px'>No results yet.</p>"

    display = rows[:show_top]
    html = """<div style='overflow-x:auto'><table>
    <tr>
        <th>Rank</th>
        <th>Candidate ID</th>
        <th>Score</th>
        <th>Skill Fit</th>
        <th>Career</th>
        <th>Availability</th>
        <th>Reasoning</th>
    </tr>"""

    for r in display:
        f = r["features"]
        sr = r["score_result"]
        skill_pct = int(sr["skill_component"] * 100)
        career_pct = int(sr["career_component"] * 100)
        avail_pct = int(sr["availability_factor"] * 100)
        bar = _score_bar_html(r["score"])
        reasoning_short = r["reasoning"][:100] + ("..." if len(r["reasoning"]) > 100 else "")

        html += f"""<tr>
        <td><span class="rank-badge">#{r['rank']}</span></td>
        <td><code style='color:#a5b4fc;font-size:11px'>{r['candidate_id']}</code></td>
        <td>{bar}<strong style='color:#e0e7ff'>{r['score']}</strong></td>
        <td><span style='color:#a5b4fc'>{skill_pct}%</span></td>
        <td><span style='color:#c4b5fd'>{career_pct}%</span></td>
        <td><span style='color:#6ee7b7'>{avail_pct}%</span></td>
        <td style='font-size:11px;color:rgba(255,255,255,0.65);max-width:260px'>{reasoning_short}</td>
        </tr>"""

    html += "</table></div>"
    return html


def build_stats_html(rows, timings, n_total):
    if not rows:
        return ""
    scores = [r["score"] for r in rows]
    top1 = rows[0]
    f = top1["features"]
    html = f"""
    <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px'>
        <div class="stat-card">
            <div style='font-size:28px;font-weight:700;color:#a5b4fc'>{n_total:,}</div>
            <div style='color:rgba(255,255,255,0.5);font-size:12px;margin-top:4px'>Candidates Ranked</div>
        </div>
        <div class="stat-card">
            <div style='font-size:28px;font-weight:700;color:#34d399'>{timings['total']}s</div>
            <div style='color:rgba(255,255,255,0.5);font-size:12px;margin-top:4px'>Total Runtime</div>
        </div>
        <div class="stat-card">
            <div style='font-size:28px;font-weight:700;color:#f9a8d4'>{scores[0]:.4f}</div>
            <div style='color:rgba(255,255,255,0.5);font-size:12px;margin-top:4px'>Top-1 Score</div>
        </div>
        <div class="stat-card">
            <div style='font-size:28px;font-weight:700;color:#fbbf24'>{len(rows)}</div>
            <div style='color:rgba(255,255,255,0.5);font-size:12px;margin-top:4px'>Candidates Selected</div>
        </div>
    </div>
    <div style='background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.25);border-radius:10px;padding:16px;margin-bottom:16px'>
        <div style='color:#34d399;font-weight:600;margin-bottom:8px'>✅ Pipeline Complete</div>
        <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:8px;font-size:12px;color:rgba(255,255,255,0.6)'>
            <span>Feature Extraction: <strong style='color:white'>{timings['feature_extraction']}s</strong></span>
            <span>Consistency Audit: <strong style='color:white'>{timings['consistency_audit']}s</strong></span>
            <span>Scoring + Selection: <strong style='color:white'>{timings['scoring_selection']}s</strong></span>
            <span>Reasoning: <strong style='color:white'>{timings['reasoning']}s</strong></span>
        </div>
    </div>
    """
    return html


def build_csv_download(rows):
    """Write submission.csv to a temp file and return path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False,
        encoding="utf-8", newline=""
    )
    writer = csv.DictWriter(tmp, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    writer.writeheader()
    for r in rows:
        writer.writerow({
            "candidate_id": r["candidate_id"],
            "rank": r["rank"],
            "score": r["score"],
            "reasoning": r["reasoning"],
        })
    tmp.close()
    return tmp.name


# ── Sample data ───────────────────────────────────────────────────────────────

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "sample_candidates.json")

def load_sample_content():
    if os.path.exists(SAMPLE_PATH):
        return open(SAMPLE_PATH, encoding="utf-8").read()
    return None


# ── Main handler ──────────────────────────────────────────────────────────────

_last_rows = []

def run_from_upload(file_obj):
    global _last_rows
    if file_obj is None:
        return (
            "<p style='color:#fbbf24;text-align:center;padding:30px'>⚠️ Please upload a candidates file first.</p>",
            "",
            None,
        )
    try:
        content = open(file_obj.name, encoding="utf-8").read()
        candidates = iter_candidates_str(content)
        if not candidates:
            return (
                "<p style='color:#f87171;text-align:center;padding:30px'>❌ Could not parse any candidates from the file.</p>",
                "",
                None,
            )
        rows, timings = run_pipeline(candidates)
        _last_rows = rows
        stats_html = build_stats_html(rows, timings, len(candidates))
        table_html = build_table_html(rows, show_top=10)
        csv_path = build_csv_download(rows)
        return stats_html + table_html, "", csv_path
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return f"<p style='color:#f87171;padding:20px'>❌ Error: {e}<br><pre style='font-size:11px;opacity:0.6'>{tb[:800]}</pre></p>", "", None


def run_sample():
    global _last_rows
    content = load_sample_content()
    if content is None:
        return (
            "<p style='color:#fbbf24;text-align:center;padding:30px'>⚠️ Sample file not found in Space.</p>",
            "",
            None,
        )
    try:
        candidates = iter_candidates_str(content)
        rows, timings = run_pipeline(candidates)
        _last_rows = rows
        stats_html = build_stats_html(rows, timings, len(candidates))
        table_html = build_table_html(rows, show_top=10)
        csv_path = build_csv_download(rows)
        return stats_html + table_html, f"✅ Ran on {len(candidates)} sample candidates", csv_path
    except Exception as e:
        import traceback
        return f"<p style='color:#f87171;padding:20px'>❌ Error: {e}</p>", "", None


def show_more_results(n):
    if not _last_rows:
        return "<p style='color:rgba(255,255,255,0.4);text-align:center'>Run the pipeline first.</p>"
    return build_table_html(_last_rows, show_top=int(n))


# ── Build UI ──────────────────────────────────────────────────────────────────

HEADER_MD = """
<div class="header-box">
    <div style="font-size:13px;letter-spacing:0.12em;text-transform:uppercase;color:rgba(165,180,252,0.7);margin-bottom:8px">
        🏆 India Runs Data & AI Challenge · Hackathon
    </div>
    <h1 style="font-size:36px;font-weight:700;background:linear-gradient(135deg,#a5b4fc,#f9a8d4,#6ee7b7);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 10px">
        Redrob Candidate Ranker
    </h1>
    <p style="color:rgba(255,255,255,0.55);font-size:15px;margin:0;max-width:600px;margin:0 auto">
        Discovers the <strong style="color:rgba(255,255,255,0.85)">top 100 ML/AI engineers</strong> from 100,000 profiles
        in under 37 seconds — no LLM, no GPU, fully explainable.
    </p>
</div>
"""

HOW_IT_WORKS = """
<div class="panel">
<h3 style="color:#a5b4fc;margin-top:0;font-size:15px;letter-spacing:0.05em">⚙️ HOW IT WORKS — 7-Stage Pipeline</h3>
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;font-size:12px">
    <div>
        <div style="color:#6ee7b7;font-weight:600">A + B</div>
        <div style="color:rgba(255,255,255,0.6);margin-top:2px">Ingest → 35 features<br>across 5 JD-aligned buckets</div>
    </div>
    <div>
        <div style="color:#f9a8d4;font-weight:600">C + D</div>
        <div style="color:rgba(255,255,255,0.6);margin-top:2px">Transparent scoring +<br>Isolation Forest honeypot audit</div>
    </div>
    <div>
        <div style="color:#a5b4fc;font-weight:600">E</div>
        <div style="color:rgba(255,255,255,0.6);margin-top:2px">O(N log K) heap<br>selects top-100</div>
    </div>
    <div>
        <div style="color:#fbbf24;font-weight:600">F + G</div>
        <div style="color:rgba(255,255,255,0.6);margin-top:2px">Template reasoning text<br>→ submission.csv</div>
    </div>
</div>
</div>
"""

with gr.Blocks(css=CUSTOM_CSS, title="Redrob Candidate Ranker") as demo:

    gr.HTML(HEADER_MD)
    gr.HTML(HOW_IT_WORKS)

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📂 Upload Candidates File", elem_classes=["panel-title"])
            gr.Markdown(
                "<p style='color:rgba(255,255,255,0.45);font-size:13px;margin-top:-8px'>"
                "Accepts <code>.jsonl</code>, <code>.jsonl.gz</code>, or <code>.json</code> array format</p>"
            )
            file_input = gr.File(
                label="candidates.jsonl",
                file_types=[".jsonl", ".json", ".gz"],
            )

        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Quick Demo", elem_classes=["panel-title"])
            gr.Markdown(
                "<p style='color:rgba(255,255,255,0.45);font-size:13px;margin-top:-8px'>"
                "Run on 50 sample candidates instantly</p>"
            )
            sample_btn = gr.Button("▶ Run Sample Demo", elem_classes=["secondary-btn"])

    with gr.Row():
        run_btn = gr.Button("🚀 Run Full Pipeline", elem_classes=["primary-btn"], scale=3)
        dl_btn = gr.DownloadButton(
            "⬇ Download submission.csv",
            elem_classes=["secondary-btn"],
            scale=1,
            visible=True,
        )

    status_box = gr.Markdown("")

    results_html = gr.HTML(
        "<div style='text-align:center;padding:60px;color:rgba(255,255,255,0.25)'>"
        "Upload a file and click Run Pipeline to see results</div>"
    )

    with gr.Accordion("Show More Results", open=False):
        n_slider = gr.Slider(minimum=5, maximum=100, value=10, step=5, label="Number of candidates to display")
        more_html = gr.HTML()
        n_slider.change(show_more_results, inputs=n_slider, outputs=more_html)

    # Wire up events
    run_btn.click(
        fn=run_from_upload,
        inputs=[file_input],
        outputs=[results_html, status_box, dl_btn],
    )

    sample_btn.click(
        fn=run_sample,
        inputs=[],
        outputs=[results_html, status_box, dl_btn],
    )

    gr.HTML("""
    <div style="text-align:center;padding:32px 0 16px;color:rgba(255,255,255,0.25);font-size:12px">
        Redrob Hackathon · India Runs Data & AI Challenge ·
        <span style="color:#6366f1">CPU-only · No LLM · Fully explainable</span>
    </div>
    """)


if __name__ == "__main__":
    demo.launch()
