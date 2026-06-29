"""Unit tests for feature extraction against sample candidates."""
import sys, os, json, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from features import extract_features

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "..", "India_runs_data_and_ai_challenge", "sample_candidates.json")


def load_sample():
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_extract_features_keys():
    """All expected feature keys are present in output."""
    data = load_sample()
    f = extract_features(data[0])
    required = [
        "b1_title_ml_score","b1_core_skill_score","b1_years_experience",
        "b2_production_score","b2_consulting_only_flag","b2_research_only_flag",
        "b3_location_score","b3_notice_score","b4_availability_score",
        "b4_recency_factor","b5_tenure_yoe_ratio","edu_tier_score","profile_completeness",
        "_candidate_id","_current_title","_years_exp",
    ]
    for k in required:
        assert k in f, f"Missing key: {k}"


def test_scores_in_range():
    """All numeric scores are in [0,1]."""
    data = load_sample()
    bounded = [
        "b1_title_ml_score","b1_core_skill_score","b1_core_skill_count_norm",
        "b2_production_score","b3_location_score","b3_notice_score",
        "b4_availability_score","b4_recency_factor","edu_tier_score","profile_completeness",
    ]
    for cand in data[:10]:
        f = extract_features(cand)
        for k in bounded:
            v = float(f.get(k,0))
            assert 0.0 <= v <= 1.0, f"{k}={v} out of range for {cand['candidate_id']}"


def test_notice_period_plateau():
    """Notice period <=30 days should score 1.0."""
    from features import _b3
    fake = {"profile":{"location":"Pune","country":"India"},"redrob_signals":{"notice_period_days":15,"willing_to_relocate":False,"preferred_work_mode":"hybrid"}}
    b = _b3(fake)
    assert b["notice_score"] == 1.0, "<=30d notice should score 1.0"

    fake["redrob_signals"]["notice_period_days"] = 90
    b = _b3(fake)
    assert b["notice_score"] < 0.65, "90d notice should score below 0.65"


def test_availability_decay():
    """Inactive candidate should have low recency_factor."""
    import math
    from features import _b4
    fake = {"redrob_signals":{
        "last_active_date":"2025-01-01","open_to_work_flag":False,
        "recruiter_response_rate":0.1,"avg_response_time_hours":100,
        "interview_completion_rate":0.5,"applications_submitted_30d":0,
    }}
    b = _b4(fake)
    assert b["recency_factor"] < 0.01, "6-month-old activity should have near-zero recency"


def test_consulting_only_flag():
    """All consulting company roles -> consulting_only_flag True."""
    from features import _b2
    fake = {"career_history":[
        {"company":"TCS","title":"Engineer","description":"consulting work","industry":"IT Services",
         "duration_months":24,"is_current":True},
        {"company":"Infosys","title":"Sr Engineer","description":"consulting","industry":"IT Services",
         "duration_months":36,"is_current":False},
    ]}
    b = _b2(fake)
    assert b["consulting_only_flag"] is True


def test_full_pipeline_sample():
    """End-to-end: extract -> score -> top selection on sample data."""
    from scoring import compute_score
    from consistency_audit import compute_suspicion
    from select_top_k import select_top_k

    data = load_sample()
    feature_list = [extract_features(c) for c in data]

    def stream(fl):
        for f in fl:
            susp = compute_suspicion(f, None)
            sr   = compute_score(f, susp)
            yield {"candidate_id":f["_candidate_id"],"score":sr["score"],
                   "features":f,"score_result":sr,"suspicion_result":susp}

    top = select_top_k(stream(feature_list), k=10)
    assert len(top) == 10
    scores = [x["score"] for x in top]
    assert scores == sorted(scores, reverse=True), "Top-K must be sorted descending"


if __name__ == "__main__":
    import traceback
    tests = [test_extract_features_keys, test_scores_in_range, test_notice_period_plateau,
             test_availability_decay, test_consulting_only_flag, test_full_pipeline_sample]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} tests passed")
