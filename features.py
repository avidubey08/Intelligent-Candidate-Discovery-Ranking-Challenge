"""
features.py — Stage B: Feature Extraction
Extracts ~35 numeric features across 5 buckets per candidate.
All feature names are stable — scoring.py and reasoning.py reference them by name.
"""
import math
from datetime import date, datetime

ML_TITLE_KEYWORDS = {
    "machine learning","ml engineer","ml scientist","ai engineer","data scientist",
    "research scientist","applied scientist","nlp engineer","computer vision",
    "deep learning","llm","search engineer","ranking engineer","recommendation",
    "recsys","applied ml","applied ai","mlops","ml platform","ml infra",
    "ml researcher","ai researcher","data science","applied research",
}

CORE_ML_SKILLS = {
    "python","machine learning","deep learning","pytorch","tensorflow","transformers",
    "llm","nlp","natural language processing","recommendation systems","ranking",
    "search","retrieval","vector database","embeddings","rag","fine-tuning",
    "scikit-learn","xgboost","lightgbm","neural network","bert","gpt","hugging face",
    "mlops","model serving","feature engineering","a/b testing","experimentation",
    "elasticsearch","faiss","pinecone","milvus","weaviate","spark","kafka","airflow",
    "kubeflow","mlflow","computer vision","image classification","object detection",
    "speech recognition","generative ai","diffusion models","reinforcement learning",
    "causal inference","fine-tuning llms","lora","qlora","gans","statistical modeling",
    "tts","bentoml","weights & biases",
}

PRODUCTION_TERMS = {
    "shipped","production","deployed","launched","released","live","real users",
    "scale","scaled","scaling","docker","kubernetes","k8s","ci/cd","pipeline","mlops",
    "latency","throughput","optimized","optimization","performance","traffic","qps",
    "rps","rollback","monitoring","alerting","on-call","oncall","incident","sla","slo",
    "uptime","reliability","a/b test","a/b testing","experiment","experimentation",
    "canary","shadow","feature flag","recommendation","ranking","retrieval","search",
    "recsys","embedding","index","serving","inference","model serving","real-time",
    "batch","streaming","online learning","cross-functional","product","stakeholder",
    "end-to-end","owner","ownership","led","lead","architected","designed",
}

CONSULTING_COMPANIES = {
    "tcs","tata consultancy","infosys","wipro","accenture","cognizant","capgemini",
    "hcl","tech mahindra","mphasis","hexaware","niit technologies","mastech",
    "kforce","l&t infotech","ltimindtree","mindtree",
}

TIER1_INDIA_CITIES = {
    "pune","noida","hyderabad","mumbai","delhi","bengaluru","bangalore","chennai",
    "gurgaon","gurugram","ncr","new delhi","navi mumbai","thane",
}

PROFICIENCY_WEIGHTS = {"beginner":0.25,"intermediate":0.55,"advanced":0.80,"expert":1.00}
REFERENCE_DATE = date(2026, 6, 28)


def _parse_date(s):
    if not s: return None
    try: return datetime.strptime(str(s)[:10],"%Y-%m-%d").date()
    except: return None

def _days_since(d):
    if d is None: return 9999.0
    return max(0.0,(REFERENCE_DATE-d).days)

def _tl(t):
    return str(t).lower().strip() if t else ""

def _hits(text,terms):
    return sum(1 for t in terms if t in text)

def _title_ml(title):
    t=_tl(title)
    if any(k in t for k in ML_TITLE_KEYWORDS): return 1.0
    if any(k in t for k in {"data engineer","analytics engineer","platform engineer",
                              "software engineer","backend engineer","research engineer"}): return 0.35
    return 0.0


def _b1(c):
    skills=c.get("skills",[]) or []
    profile=c.get("profile",{}) or {}
    title_score=_title_ml(profile.get("current_title",""))
    assess=c.get("redrob_signals",{}).get("skill_assessment_scores",{}) or {}
    wscore=0.0; cnt=0
    for sk in skills:
        name=_tl(sk.get("name",""))
        prof=_tl(sk.get("proficiency","beginner"))
        end=max(0,sk.get("endorsements",0) or 0)
        dur=max(0,sk.get("duration_months",0) or 0)
        pw=PROFICIENCY_WEIGHTS.get(prof,0.25)
        ef=math.log1p(end)/math.log1p(100)
        df=math.log1p(dur)/math.log1p(60)
        ss=pw*(0.4+0.35*ef+0.25*df)
        for ak,av in assess.items():
            if _tl(ak)==name: ss*=0.8+0.4*(av/100); break
        if name in CORE_ML_SKILLS:
            cnt+=1; wscore+=ss
    gh=c.get("redrob_signals",{}).get("github_activity_score",-1) or -1
    return {
        "title_ml_score":title_score,
        "core_skill_score":min(1.0,wscore/10.0),
        "core_skill_count":cnt,
        "core_skill_count_norm":min(1.0,cnt/12.0),
        "github_activity":max(0.0,gh/100.0) if gh>=0 else 0.0,
        "years_experience":float(profile.get("years_of_experience",0) or 0),
    }


def _b2(c):
    history=c.get("career_history",[]) or []
    zero={
        "production_term_hits":0,"production_score":0.0,
        "consulting_only_flag":False,"consulting_penalty":0.0,
        "research_only_flag":False,"title_chasing_flag":False,
        "title_chasing_penalty":0.0,"career_company_product_ratio":0.0,
        "total_career_months":0,
    }
    if not history: return zero
    rk={"phd","research","lab","academia","university","professor","postdoc",
        "iit","iim","iisc","mit","stanford","oxford","research institute","national laboratory"}
    descs=[]; cons=0; ronly=0; prod=0; hops=0; months=0
    for r in history:
        co=_tl(r.get("company","") or ""); ti=_tl(r.get("title","") or "")
        desc=_tl(r.get("description","") or ""); ind=_tl(r.get("industry","") or "")
        dur=r.get("duration_months",0) or 0
        descs.append(desc); months+=dur
        if any(x in co for x in CONSULTING_COMPANIES): cons+=1
        in_rk=any(k in co or k in ind or k in ti for k in rk)
        has_prod=any(t in desc for t in PRODUCTION_TERMS)
        if in_rk and not has_prod: ronly+=1
        else: prod+=1
        if not r.get("is_current",False) and dur<=18: hops+=1
    txt=" ".join(descs)
    ph=_hits(txt,PRODUCTION_TERMS)
    ps=min(1.0,math.log1p(ph)/math.log1p(30))
    n=len(history)
    conf=cons==n and n>0; rof=ronly==n and n>0 and prod==0; tchase=hops>=3
    return {
        "production_term_hits":ph,"production_score":ps,
        "consulting_only_flag":conf,"consulting_penalty":0.25 if conf else 0.0,
        "research_only_flag":rof,"title_chasing_flag":tchase,
        "title_chasing_penalty":min(0.5,hops*0.12) if tchase else 0.0,
        "career_company_product_ratio":prod/max(1,n),
        "total_career_months":months,
    }


def _b3(c):
    profile=c.get("profile",{}) or {}; sig=c.get("redrob_signals",{}) or {}
    loc=_tl(profile.get("location","") or ""); country=_tl(profile.get("country","") or "")
    reloc=bool(sig.get("willing_to_relocate",False))
    notice=int(sig.get("notice_period_days",30) or 30)
    mode=_tl(sig.get("preferred_work_mode","") or "")
    india=country=="india"; tier1=any(ct in loc for ct in TIER1_INDIA_CITIES)
    if india and tier1: ls=1.0
    elif india and reloc: ls=0.85
    elif india: ls=0.65
    elif reloc: ls=0.45
    else: ls=0.20
    if notice<=30: ns=1.0
    elif notice<=60: ns=1.0-0.35*((notice-30)/30)
    elif notice<=90: ns=0.65-0.25*((notice-60)/30)
    else: ns=max(0.10,0.40-0.15*((notice-90)/30))
    ms={"flexible":1.0,"hybrid":0.95,"onsite":0.85,"remote":0.70}.get(mode,0.80)
    return {
        "location_score":ls,"in_india":float(india),"in_tier1_city":float(tier1),
        "willing_to_relocate":float(reloc),"notice_period_days":notice,
        "notice_score":ns,"work_mode_score":ms,
    }


def _b4(c):
    sig=c.get("redrob_signals",{}) or {}
    days=_days_since(_parse_date(sig.get("last_active_date")))
    rec=math.exp(-math.log(2)/30.0*days)
    otw=bool(sig.get("open_to_work_flag",False))
    rr=max(0.0,min(1.0,float(sig.get("recruiter_response_rate",0.5) or 0.5)))
    arh=max(0.5,float(sig.get("avg_response_time_hours",24.0) or 24.0))
    rtf=max(0.0,1.0-math.log(arh)/math.log(336.0))
    ir=max(0.0,min(1.0,float(sig.get("interview_completion_rate",0.7) or 0.7)))
    apps=int(sig.get("applications_submitted_30d",0) or 0)
    avail=max(0.05,min(1.0,0.35*rec+0.25*rr+0.20*(1.0 if otw else 0.5)+0.12*ir+0.08*rtf))
    return {
        "days_inactive":days,"recency_factor":rec,"open_to_work":float(otw),
        "recruiter_response_rate":rr,"avg_response_time_hours":arh,
        "response_time_factor":rtf,"interview_completion_rate":ir,
        "apps_30d":apps,"availability_score":avail,
    }


def _b5(c):
    history=c.get("career_history",[]) or []
    skills=c.get("skills",[]) or []
    profile=c.get("profile",{}) or {}
    yoe=float(profile.get("years_of_experience",0) or 0)*12.0
    maxr=max((r.get("duration_months",0) or 0 for r in history),default=0)
    tot=sum(r.get("duration_months",0) or 0 for r in history)
    ratio=tot/max(1.0,yoe); imp=maxr>max(1,yoe*1.1)
    es=0; mm=0
    for sk in skills:
        p=_tl(sk.get("proficiency",""))
        dur=sk.get("duration_months",0) or 0
        end=sk.get("endorsements",0) or 0
        if p=="expert" and dur<12: es+=1
        if p in ("advanced","expert") and dur<6: mm+=1
        if dur>0 and end>0 and end/dur>5.0: mm+=1
    return {
        "tenure_yoe_ratio":ratio,"impossible_tenure_flag":float(imp),
        "expert_short_skill_count":es,
        "proficiency_duration_mismatch_rate":mm/max(1,len(skills)),
        "n_skills":len(skills),"n_roles":len(history),
        "claimed_career_months":tot,"stated_yoe_months":yoe,
    }


def _edu(c):
    ed=c.get("education",[]) or []
    if not ed: return {"edu_tier_score":0.5,"edu_stem_flag":False}
    tm={"tier_1":1.0,"tier_2":0.80,"tier_3":0.60,"tier_4":0.40,"unknown":0.50}
    sf={"computer science","cs","it","information technology","electrical","electronics",
        "mathematics","statistics","machine learning","data science","ai",
        "information systems","engineering","physics","computational"}
    best=0.5; stem=False
    for e in ed:
        t=tm.get(_tl(e.get("tier","unknown") or "unknown"),0.5)
        best=max(best,t)
        if any(s in _tl(e.get("field_of_study","") or "") for s in sf): stem=True
        deg=_tl(e.get("degree","") or "")
        if "phd" in deg or "ph.d" in deg or "doctor" in deg: best=min(1.0,best+0.10)
        elif any(x in deg for x in ("m.tech","m.e.","m.s","master")): best=min(1.0,best+0.05)
    return {"edu_tier_score":best,"edu_stem_flag":stem}


def extract_features(candidate: dict) -> dict:
    """
    Extract all features for a single candidate. Returns flat dict with prefixed keys:
      b1_* skill/role fit | b2_* career substance | b3_* logistics
      b4_* availability   | b5_* consistency audit | edu_* education
      _*   metadata for reasoning (not used in scoring)
    """
    b1=_b1(candidate); b2=_b2(candidate); b3=_b3(candidate)
    b4=_b4(candidate); b5=_b5(candidate); edu=_edu(candidate)
    profile=candidate.get("profile",{}) or {}
    sig=candidate.get("redrob_signals",{}) or {}
    f={}
    for k,v in b1.items(): f[f"b1_{k}"]=v
    for k,v in b2.items(): f[f"b2_{k}"]=v
    for k,v in b3.items(): f[f"b3_{k}"]=v
    for k,v in b4.items(): f[f"b4_{k}"]=v
    for k,v in b5.items(): f[f"b5_{k}"]=v
    f["edu_tier_score"]=edu["edu_tier_score"]
    f["edu_stem_flag"]=float(edu["edu_stem_flag"])
    f["profile_completeness"]=float(sig.get("profile_completeness_score",50) or 50)/100.0
    f["verified_email"]=float(bool(sig.get("verified_email",False)))
    f["verified_phone"]=float(bool(sig.get("verified_phone",False)))
    f["linkedin_connected"]=float(bool(sig.get("linkedin_connected",False)))
    f["saved_by_recruiters_30d"]=min(1.0,int(sig.get("saved_by_recruiters_30d",0) or 0)/10.0)
    f["_candidate_id"]=candidate.get("candidate_id","")
    f["_current_title"]=profile.get("current_title","") or ""
    f["_current_company"]=profile.get("current_company","") or ""
    f["_location"]=profile.get("location","") or ""
    f["_years_exp"]=float(profile.get("years_of_experience",0) or 0)
    f["_notice_period"]=int(sig.get("notice_period_days",30) or 30)
    f["_response_rate"]=float(sig.get("recruiter_response_rate",0) or 0)
    f["_open_to_work"]=bool(sig.get("open_to_work_flag",False))
    f["_last_active"]=sig.get("last_active_date","") or ""
    f["_core_skill_count"]=int(b1["core_skill_count"])
    f["_production_hits"]=int(b2["production_term_hits"])
    f["_current_company_consulting"]=bool(b2["consulting_only_flag"])
    return f
