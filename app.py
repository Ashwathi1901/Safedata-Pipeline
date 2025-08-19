# app.py — SafeData Pipeline (Gorgeous UI Edition)

import os, io, yaml
import streamlit as st
import pandas as pd
from modules import risk as risk_mod
from modules import privacy as priv
from modules import utility as util
from modules import reporting as rep
from modules import compliance as comp

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SafeData Pipeline", layout="wide", page_icon="🔐")

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES (Gradient + Glassmorphism + Stepper + Pretty Widgets)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Page background (soft gradient) */
.main, .block-container {
  background: linear-gradient(135deg, #e0f2fe 0%, #f1f5f9 100%) !important;
}

/* Hide Streamlit default header/footer gaps slightly for tighter look */
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }

/* Top brand bar */
.brand-wrap {
  background: linear-gradient(120deg, #0ea5e9, #22c55e);
  border-radius: 18px;
  padding: 18px 22px;
  color: #ffffff;
  box-shadow: 0 10px 30px rgba(14,165,233,.25);
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
}
.brand-title {
  font-size: 1.35rem;
  font-weight: 700;
  letter-spacing: .2px;
}
.brand-sub {
  opacity: .95;
  font-size: .95rem;
}

/* Glass cards */
.glass {
  background: rgba(255,255,255,0.82);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  border: 1px solid rgba(255,255,255,0.55);
  border-radius: 18px;
  box-shadow: 0 10px 30px rgba(2,6,23,.08);
  padding: 18px 18px 6px 18px;
  margin-bottom: 18px;
}

/* Section titles */
.h2 {
  font-size: 1.2rem; 
  font-weight: 700;
  color: #0f172a;
  display: inline-flex; 
  gap: 10px; 
  align-items: center;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b1220, #0f172a);
  color: #e2e8f0;
  border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .sidebar-title {
  font-weight: 700; 
  margin: 8px 0 12px 0; 
  font-size: 1.05rem;
}

/* Stepper pills */
.stepper {
  display: grid;
  gap: 8px;
}
.step-pill {
  padding: 10px 12px;
  border-radius: 12px;
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(148,163,184,0.25);
  transition: all .2s ease;
  display: flex; 
  gap: 10px; 
  align-items: center;
  font-weight: 600;
}
.step-pill:hover { transform: translateY(-1px); background: rgba(255,255,255,0.08); }
.step-active { background: linear-gradient(120deg, rgba(14,165,233,.22), rgba(34,197,94,.22)); border-color: rgba(59,130,246,.35); }

/* Metric tweaks */
[data-testid="stMetricValue"] { font-weight: 800; }

/* Sticky bottom action bar */
.actionbar {
  position: sticky;
  bottom: 0;
  z-index: 20;
  background: rgba(255,255,255,0.8);
  backdrop-filter: blur(6px);
  border: 1px solid rgba(0,0,0,0.05);
  border-radius: 14px;
  padding: 12px 14px;
  margin-top: 10px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

/* Tabs underline accent */
.stTabs [role="tablist"] { border-bottom: 2px solid #0ea5e9; }

/* Dataframe rounded corners */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* Buttons */
button[kind="secondary"] { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR (Stepper + Config Save/Load)
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("<div class='sidebar-title'>🔐 SafeData Pipeline</div>", unsafe_allow_html=True)

steps = ["Upload", "Risk", "Protect", "Utility", "Compliance", "Report"]
icons = {
    "Upload": "📂", "Risk": "⚠️", "Protect": "🛡️",
    "Utility": "📊", "Compliance": "✅", "Report": "📝"
}

# We render a custom stepper but still use radio underneath for state
st.sidebar.markdown("<div class='sidebar-title'>Pipeline Steps</div>", unsafe_allow_html=True)
selected = st.sidebar.radio(" ", steps, index=0, format_func=lambda x: f"{icons[x]} {x}")

with st.sidebar.expander("⚙️ Save / Load Config", expanded=False):
    if "config" not in st.session_state: st.session_state.config = {}
    cfg = st.session_state.config
    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 Save"):
            cfg_bytes = io.BytesIO(yaml.safe_dump(cfg).encode("utf-8"))
            st.download_button("Download config.yaml", data=cfg_bytes.getvalue(),
                               file_name="config.yaml", mime="text/yaml", use_container_width=True)
    with c2:
        cfg_file = st.file_uploader("Load config.yaml", type=["yaml","yml"], label_visibility="collapsed")
        if cfg_file is not None:
            st.session_state.config = yaml.safe_load(cfg_file.read()) or {}
            st.success("Config loaded ✓")

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "df_real" not in st.session_state: st.session_state.df_real = None
if "df_anon" not in st.session_state: st.session_state.df_anon = None
if "df_protected" not in st.session_state: st.session_state.df_protected = None
if "risk" not in st.session_state: st.session_state.risk = {}
if "checklist" not in st.session_state: st.session_state.checklist = comp.default_checklist()

def load_csv(uploaded):
    if uploaded is None:
        return None
    try:
        return pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# BRAND BAR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="brand-wrap">
  <div class="brand-title">🔐 SafeData Pipeline</div>
  <div class="brand-sub">Assess risk • Enhance privacy • Preserve utility • Prove compliance • Export reports</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP: UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
if selected == "Upload":
    st.markdown(f"<div class='glass'><div class='h2'>📂 Upload Datasets</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        real_file = st.file_uploader("Real-ID dataset (for linkage risk benchmarking)", type=["csv"], key="real_upl")
        if real_file:
            st.session_state.df_real = load_csv(real_file)
            if st.session_state.df_real is not None:
                st.dataframe(st.session_state.df_real.head(), use_container_width=True)
    with c2:
        anon_file = st.file_uploader("Anonymous dataset (candidate for sharing)", type=["csv"], key="anon_upl")
        if anon_file:
            st.session_state.df_anon = load_csv(anon_file)
            if st.session_state.df_anon is not None:
                st.dataframe(st.session_state.df_anon.head(), use_container_width=True)
    st.info("💡 Tip: Include likely quasi-identifiers such as Age, Gender, City, Pincode/Zip, etc.")
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP: RISK
# ─────────────────────────────────────────────────────────────────────────────
if selected == "Risk":
    st.markdown(f"<div class='glass'><div class='h2'>⚠️ Risk Assessment</div>", unsafe_allow_html=True)

    if st.session_state.df_real is None or st.session_state.df_anon is None:
        st.warning("Please upload both datasets first.")
    else:
        df_real, df_anon = st.session_state.df_real, st.session_state.df_anon
        st.caption("Suggested quasi-identifiers:")
        st.write(getattr(risk_mod, "QUASI_ID_SUGGESTIONS", []))

        default_quasi = [c for c in getattr(risk_mod, "QUASI_ID_SUGGESTIONS", []) if c in df_anon.columns]
        quasi = st.multiselect("Select quasi-identifiers to test linkage risk:", 
                               options=df_anon.columns.tolist(), default=default_quasi)

        if len(quasi) == 0:
            st.info("Select at least one quasi-identifier to run assessment.")
        else:
            try:
                with st.spinner("Running re-identification risk simulation..."):
                    risk_score, row_scores, nn_idx, dists = risk_mod.simulate_reidentification_risk(
                        df_anon, df_real, quasi
                    )
                c1, c2, c3 = st.columns(3)
                c1.metric("Estimated Linkage Risk", f"{risk_score:.3f}")
                c2.metric("Rows (Anon)", len(df_anon))
                c3.metric("Rows (Real)", len(df_real))

                st.line_chart(pd.Series(row_scores, name="Row-level Risk"))
                st.session_state.risk = {"risk_score": risk_score, "quasi": quasi}
                st.success("Risk assessment completed ✓")
            except Exception as e:
                st.error(f"Risk estimation failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP: PROTECT
# ─────────────────────────────────────────────────────────────────────────────
if selected == "Protect":
    st.markdown(f"<div class='glass'><div class='h2'>🛡️ Privacy Enhancement</div>", unsafe_allow_html=True)

    if st.session_state.df_anon is None:
        st.warning("Upload an anonymous dataset first.")
    else:
        df = st.session_state.df_anon
        # Smart suggestions
        sugg = priv.smart_suggest(df)
        st.caption("Smart suggestions (you can customize below):")
        st.write(sugg)

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        sdc_cols = st.multiselect("Suppress rare categories in columns:", options=df.columns.tolist(),
                                  default=sugg.get("sdc_cols", [])[:5])
        gen_cols = st.multiselect("Generalize (bin) numeric columns:", options=numeric_cols,
                                  default=[c for c in sugg.get("generalize_cols", []) if c in numeric_cols][:5])
        dp_opts = st.multiselect("Add DP-style noise to numeric columns:", options=numeric_cols,
                                 default=[c for c in sugg.get("dp_cols", []) if c in numeric_cols][:5])
        epsilon = st.slider("DP epsilon (lower = more privacy, typical 0.5–2.0)",
                            min_value=0.1, max_value=5.0, step=0.1, value=float(sugg.get("dp_epsilon", 1.0)))
        method = st.selectbox("Synthetic data generation (optional):", ["None", "Lightweight sampler"])

        with st.spinner("Applying protections..."):
            df_prot = df.copy()

            # SDC suppression (categorical k-anon style)
            df_prot = priv.sdc_suppress(df_prot, sdc_cols, threshold=5)

            # Generalization (binning) for selected numeric columns
            if len(gen_cols) > 0:
                df_prot = priv.generalize_numeric(df_prot, gen_cols, bins=10)

            # DP noise (single pass AFTER generalization)
            if len(dp_opts) > 0:
                # Guard: ensure still numeric after possible generalization
                dp_final = [c for c in dp_opts if c in df_prot.columns and pd.api.types.is_numeric_dtype(df_prot[c])]
                if len(dp_final) > 0:
                    df_prot = priv.add_dp_noise(df_prot, dp_final, epsilon=epsilon, sensitivity=1.0)

            # Optional synthetic sampling
            if method == "Lightweight sampler":
                df_prot = priv.synthetic_sample(df_prot, n=len(df))

        st.session_state.df_protected = df_prot
        st.success("Protection applied ✓")
        st.dataframe(df_prot.head(), use_container_width=True)

        st.download_button(
            "⬇️ Download protected.csv",
            df_prot.to_csv(index=False),
            file_name="protected.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP: UTILITY
# ─────────────────────────────────────────────────────────────────────────────
if selected == "Utility":
    st.markdown(f"<div class='glass'><div class='h2'>📊 Utility Measurement</div>", unsafe_allow_html=True)
    if st.session_state.df_anon is None or st.session_state.df_protected is None:
        st.warning("Run protection first.")
    else:
        before, after = st.session_state.df_anon, st.session_state.df_protected
        tab1, tab2, tab3 = st.tabs(["📈 Overall Stats", "📉 Distribution Drift", "🤖 ML Utility"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Before (original anonymous)**")
                st.dataframe(util.basic_stats(before), use_container_width=True)
            with c2:
                st.write("**After (protected)**")
                st.dataframe(util.basic_stats(after), use_container_width=True)

        with tab2:
            drift = util.distribution_drift(before, after)
            st.write("**Feature-wise Drift**")
            st.dataframe(drift, use_container_width=True)

        with tab3:
            target = st.selectbox("Optional: select target column for quick ML utility check (classification):",
                                  options=["<none>"] + before.columns.tolist(), index=0)
            if target != "<none>":
                try:
                    util_results = util.model_utility_check(before, after, target)
                    st.write("**Model utility (LogReg / RF fallback)**")
                    st.dataframe(util_results, use_container_width=True)
                except Exception as e:
                    st.error(f"Utility check failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP: COMPLIANCE
# ─────────────────────────────────────────────────────────────────────────────
if selected == "Compliance":
    st.markdown(f"<div class='glass'><div class='h2'>✅ Compliance Checklist (DPDP/GDPR)</div>", unsafe_allow_html=True)

    df_chk = st.session_state.checklist
    edit = st.data_editor(df_chk, num_rows="dynamic", key="chkedit", use_container_width=True)
    st.session_state.checklist = edit
    score = comp.score(edit)
    st.metric("Checklist Completion", f"{int(score*100)}%")

    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP: REPORT
# ─────────────────────────────────────────────────────────────────────────────
if selected == "Report":
    st.markdown(f"<div class='glass'><div class='h2'>📝 Reporting & Export</div>", unsafe_allow_html=True)

    summary = {
        "quasi_ids": st.session_state.get("risk", {}).get("quasi", []),
        "risk_score": st.session_state.get("risk", {}).get("risk_score", None),
        "rows_before": int(st.session_state.df_anon.shape[0]) if st.session_state.df_anon is not None else None,
        "rows_after": int(st.session_state.df_protected.shape[0]) if st.session_state.df_protected is not None else None,
    }

    util_stats = {}
    if st.session_state.df_anon is not None:
        util_stats["Stats BEFORE"] = util.basic_stats(st.session_state.df_anon)
    if st.session_state.df_protected is not None:
        util_stats["Stats AFTER"] = util.basic_stats(st.session_state.df_protected)

    # Dashboard metrics row
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows Before", summary["rows_before"])
    c2.metric("Rows After", summary["rows_after"])
    c3.metric("Risk Score", f"{summary['risk_score']:.3f}" if summary["risk_score"] is not None else "N/A")

    # Save + export
    html_path = os.path.abspath("safedata_report.html")
    pdf_path = os.path.abspath("safedata_report.pdf")
    comp_df = st.session_state.checklist
    risk_summary = st.session_state.get("risk", {})

    try:
        rep.save_html_report(html_path, summary, risk_summary, util_stats, comp_df)
        pdf_file = rep.try_make_pdf(html_path, pdf_path)
        with open(html_path, "rb") as f:
            st.download_button("⬇️ Download Report (HTML)", data=f.read(),
                               file_name="safedata_report.html", mime="text/html",
                               use_container_width=True)
        if pdf_file and os.path.exists(pdf_file):
            with open(pdf_file, "rb") as f:
                st.download_button("⬇️ Download Report (PDF)", data=f.read(),
                                   file_name="safedata_report.pdf", mime="application/pdf",
                                   use_container_width=True)
        else:
            st.info("PDF export uses a simple fallback. Install 'reportlab' for advanced layout.")
    except Exception as e:
        st.error(f"Report generation failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# STICKY ACTION BAR (Helpful hints / quick state glance)
# ─────────────────────────────────────────────────────────────────────────────
with st.container():
    st.markdown("""
    <div class="actionbar">
      <div>✨ Pro tip: Use **Utility → Drift** to tune epsilon/generalization until stats are stable.</div>
      <div>📦 Rows: <b>{rows_before}</b> → <b>{rows_after}</b> &nbsp;&nbsp; | &nbsp;&nbsp; 🔒 Risk: <b>{risk}</b></div>
    </div>
    """.format(
        rows_before=(len(st.session_state.df_anon) if st.session_state.df_anon is not None else "—"),
        rows_after=(len(st.session_state.df_protected) if st.session_state.df_protected is not None else "—"),
        risk=(f"{st.session_state.get('risk',{}).get('risk_score'):.3f}"
              if st.session_state.get("risk",{}).get("risk_score") is not None else "—")
    ), unsafe_allow_html=True)
