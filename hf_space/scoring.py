"""
scoring.py — Stage C: Transparent Weighted Scoring
Formula:
  base_fit = w_skill*skill + w_career*career + w_logistics*logistics + w_edu*edu
  final_score = base_fit * availability_factor * (1 - SUSPICION_MAX_PENALTY*suspicion)
All weights documented. No opaque model. Every weight has a JD-sourced rationale.
"""
import math

W_SKILL      = 0.40   # title-function match + weighted core ML skills
W_CAREER     = 0.35   # production language evidence + product-company ratio
W_LOGISTICS  = 0.15   # location, notice period, work mode
W_EDU        = 0.10   # education tier + STEM field

assert abs(W_SKILL+W_CAREER+W_LOGISTICS+W_EDU-1.0)<1e-9

SUSPICION_MAX_PENALTY = 0.40   # max 40% down-weight for fully suspicious candidate
RESEARCH_ONLY_FLOOR   = 0.02   # near-zero for pure research profiles (per JD)


def _skill(f):
    title =float(f.get("b1_title_ml_score",0.0) or 0.0)
    core  =float(f.get("b1_core_skill_score",0.0) or 0.0)
    github=float(f.get("b1_github_activity",0.0) or 0.0)
    yoe   =float(f.get("b1_years_experience",0.0) or 0.0)
    yoe_s =min(1.0,math.log1p(yoe)/math.log1p(12))
    return min(1.0,0.40*title+0.40*core+0.15*yoe_s+0.05*github)


def _career(f):
    if bool(f.get("b2_research_only_flag",False)): return RESEARCH_ONLY_FLOOR
    prod   =float(f.get("b2_production_score",0.0) or 0.0)
    pratio =float(f.get("b2_career_company_product_ratio",0.0) or 0.0)
    cp     =float(f.get("b2_consulting_penalty",0.0) or 0.0)
    tp     =float(f.get("b2_title_chasing_penalty",0.0) or 0.0)
    return max(0.0,min(1.0,0.60*prod+0.40*pratio-cp-tp))


def _logistics(f):
    loc   =float(f.get("b3_location_score",0.0) or 0.0)
    notice=float(f.get("b3_notice_score",0.0) or 0.0)
    mode  =float(f.get("b3_work_mode_score",0.0) or 0.0)
    return 0.50*loc+0.35*notice+0.15*mode


def _edu(f):
    tier=float(f.get("edu_tier_score",0.5) or 0.5)
    stem=float(f.get("edu_stem_flag",0.0) or 0.0)
    return 0.70*tier+0.30*stem


def compute_score(features, suspicion_result):
    """
    Compute final score for one candidate.
    Returns dict: score, base_fit, availability_factor, suspicion_score,
                  skill_component, career_component, logistics_component,
                  edu_component, top_features
    """
    sc=_skill(features); cc=_career(features)
    lc=_logistics(features); ec=_edu(features)
    base=W_SKILL*sc+W_CAREER*cc+W_LOGISTICS*lc+W_EDU*ec
    avail=float(features.get("b4_availability_score",0.5) or 0.5)
    susp=float((suspicion_result or {}).get("suspicion_score",0.0) or 0.0)
    smul=1.0-SUSPICION_MAX_PENALTY*susp
    final=max(0.0,min(1.0,base*avail*smul))
    top=sorted([
        ("skill_fit",sc,W_SKILL*sc),
        ("career_substance",cc,W_CAREER*cc),
        ("logistics",lc,W_LOGISTICS*lc),
        ("education",ec,W_EDU*ec),
        ("availability",avail,avail*0.1),
    ],key=lambda x:x[2],reverse=True)
    return {
        "score":final,"base_fit":base,"availability_factor":avail,
        "suspicion_score":susp,"suspicion_multiplier":smul,
        "skill_component":sc,"career_component":cc,
        "logistics_component":lc,"edu_component":ec,"top_features":top,
    }
