import pandas as pd

DPDP_ITEMS = [
    ("lawful_purpose_documented", "Lawful purpose documented for processing/sharing"),
    ("consent_or_legal_basis", "Consent or other legal basis recorded"),
    ("pii_identified", "PII fields identified and cataloged"),
    ("minimization_applied", "Data minimization applied (only necessary fields retained)"),
    ("privacy_techniques", "Privacy techniques applied (SDC/DP/Synthetic)"),
    ("logging_enabled", "Logging/Audit trail enabled for data transformations"),
    ("retention_policy", "Retention period defined & enforced"),
]

GDPR_ITEMS = [
    ("dpa_dpia_done", "DPA/DPIA performed when required"),
    ("dpo_contact", "Data Protection Officer contact available (if required)"),
    ("data_subject_rights", "Mechanism for data subject rights (access, rectification, erasure)"),
    ("cross_border_checks", "Cross-border transfer checks (SCCs/adequacy)"),
    ("privacy_by_design", "Privacy by design/defaults considered"),
]

def default_checklist():
    all_items = DPDP_ITEMS + GDPR_ITEMS
    return pd.DataFrame([{"key": k, "description": d, "status": False, "notes": ""} for k, d in all_items])

def score(df_checklist: pd.DataFrame):
    if df_checklist.empty:
        return 0.0
    return float(df_checklist["status"].mean())
