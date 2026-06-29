"""
consistency_audit.py — Stage D: Consistency Audit
Two-layer honeypot detection:
  Layer 1: Rule-based cross-field checks (explainable)
  Layer 2: Isolation Forest on engineered consistency features
Output: suspicion_score in [0,1] — applied as DOWN-WEIGHT in scoring.py.
"""
import numpy as np
from sklearn.ensemble import IsolationForest

TENURE_RATIO_THRESHOLD = 1.35
MISMATCH_RATE_THRESHOLD = 0.40

AUDIT_KEYS = [
    "b5_tenure_yoe_ratio",
    "b5_impossible_tenure_flag",
    "b5_expert_short_skill_count",
    "b5_proficiency_duration_mismatch_rate",
    "b5_n_skills",
    "b5_n_roles",
]


def _rules(features):
    flags=[]; score=0.0
    ratio=float(features.get("b5_tenure_yoe_ratio",1.0) or 1.0)
    if ratio>TENURE_RATIO_THRESHOLD:
        w=min(0.40,0.15*(ratio-1.0))
        flags.append((w,"Claimed career months are {:.1f}x stated YOE".format(ratio)))
        score+=w
    if float(features.get("b5_impossible_tenure_flag",0.0) or 0.0)>0.5:
        flags.append((0.35,"Single role duration exceeds total stated YOE"))
        score+=0.35
    es=int(features.get("b5_expert_short_skill_count",0) or 0)
    if es>=2:
        w=min(0.30,es*0.10)
        flags.append((w,"{} expert skills claimed with under 12 months practice".format(es)))
        score+=w
    mr=float(features.get("b5_proficiency_duration_mismatch_rate",0.0) or 0.0)
    if mr>MISMATCH_RATE_THRESHOLD:
        w=min(0.25,mr*0.5)
        flags.append((w,"Proficiency-duration mismatch rate: {:.0%}".format(mr)))
        score+=w
    return min(1.0,score), flags


def _matrix(all_features):
    return np.array(
        [[float(f.get(k,0.0) or 0.0) for k in AUDIT_KEYS] for f in all_features],
        dtype=np.float32
    )


def train_isolation_forest(all_features, contamination=0.05):
    """Train Isolation Forest on consistency features from all candidates."""
    X=_matrix(all_features)
    iso=IsolationForest(n_estimators=100,contamination=contamination,
                        max_samples="auto",random_state=42,n_jobs=-1)
    iso.fit(X)
    return iso


def compute_suspicion(features, iso_model):
    """
    Compute suspicion score [0,1] for one candidate.
    Returns: {suspicion_score, rule_score, iso_score, flags}
    """
    rule_score,flags=_rules(features)
    flag_texts=[d for (_,d) in flags]
    if iso_model is not None:
        row=np.array([[float(features.get(k,0.0) or 0.0) for k in AUDIT_KEYS]],dtype=np.float32)
        raw=float(iso_model.decision_function(row)[0])
        iso_score=max(0.0,min(1.0,(-raw+0.1)/0.5))
    else:
        iso_score=0.0
    combined=max(0.0,min(1.0,0.65*rule_score+0.35*iso_score))
    return {"suspicion_score":combined,"rule_score":rule_score,"iso_score":iso_score,"flags":flag_texts}
