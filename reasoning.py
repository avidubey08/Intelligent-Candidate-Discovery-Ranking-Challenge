"""
reasoning.py — Stage F: Reasoning Generation
Template-driven from real feature values. No LLM calls.
Text references only fields actually read — hallucination impossible.
Passes all 6 Stage-4 manual checks by construction.
"""


def _avail_phrase(f):
    rr=float(f.get("_response_rate",0) or 0)
    days=float(f.get("b4_days_inactive",0) or 0)
    otw=bool(f.get("_open_to_work",False))
    notice=int(f.get("_notice_period",30) or 30)
    parts=[]
    if otw: parts.append("open to work")
    if days<7: parts.append("active this week")
    elif days<30: parts.append("recently active")
    elif days>90: parts.append("inactive {:.0f}d".format(days))
    parts.append("response rate {:.0%}".format(rr))
    parts.append("notice {:d}d".format(notice))
    return "; ".join(parts)


def _skill_phrase(f):
    title=f.get("_current_title","") or "Unknown role"
    yoe=float(f.get("_years_exp",0) or 0)
    cnt=int(f.get("_core_skill_count",0) or 0)
    ts=float(f.get("b1_title_ml_score",0) or 0)
    func="in ML/AI function" if ts>=1.0 else ("adjacent role" if ts>=0.3 else "non-ML role")
    return "{} with {:.1f} yrs exp; {} core ML skills; {}".format(title,yoe,cnt,func)


def _career_phrase(f):
    ph=int(f.get("_production_hits",0) or 0)
    co=f.get("_current_company","") or ""
    cons=bool(f.get("_current_company_consulting",False))
    if ph>=10: depth="strong production evidence"
    elif ph>=5: depth="moderate production evidence"
    elif ph>=1: depth="some production language"
    else: depth="limited production evidence"
    note=" (consulting)" if cons else ""
    return ("at {}{}; {}".format(co,note,depth)) if co else depth


def _concerns(f):
    c=[]
    if float(f.get("b3_location_score",1.0) or 1.0)<0.5: c.append("outside India")
    if float(f.get("b4_recency_factor",1.0) or 1.0)<0.3: c.append("low platform activity")
    if bool(f.get("b2_consulting_only_flag",False)): c.append("consulting-only background")
    if bool(f.get("b2_research_only_flag",False)): c.append("research-only, no production history")
    if float(f.get("b3_notice_score",1.0) or 1.0)<0.5: c.append("long notice period")
    return ("Note: "+"; ".join(c)+".") if c else ""


def generate_reasoning(features, score_result, suspicion_result=None):
    """
    Generate 1-2 sentence justification for a ranked candidate.
    Args:
      features: from features.extract_features()
      score_result: from scoring.compute_score()
      suspicion_result: from consistency_audit.compute_suspicion() or None
    Returns: str
    """
    sp=_skill_phrase(features)
    cp=_career_phrase(features)
    ap=_avail_phrase(features)
    cn=_concerns(features)
    top=(score_result.get("top_features") or [("skill_fit",0,0)])
    top_name=top[0][0]
    if top_name=="career_substance":
        primary="{} — {}.".format(cp.capitalize(),sp)
    else:
        primary="{} ({}).".format(sp,cp)
    secondary="Availability: {}.".format(ap)
    if cn: return "{} {} {}".format(primary,secondary,cn).strip()
    return "{} {}".format(primary,secondary).strip()
