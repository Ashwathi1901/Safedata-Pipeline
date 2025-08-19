import os, io, yaml
import streamlit as st
import pandas as pd
import numpy as np
from modules import risk as risk_mod
from modules import privacy as priv
from modules import utility as util
from modules import reporting as rep
from modules import compliance as comp

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="🔐 SafeData Pipeline", layout="wide")
st.title("🔐 SafeData Pipeline")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1e293b;
        color: white;
    }
    section[data-testid="stSidebar"] .stButton>button {
        background-color: #334155;
        color: white;
        border-radius: 8px;
    }

    h1, h2, h3 {
        color: #0f172a !important;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #2563eb;
    }

    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR NAVIGATION ------------------
st.sidebar.header("📌 Pipeline Steps")
steps = {
    "Upload": "📂 Upload",
    "Risk": "⚠️ Risk",
    "Protect": "🛡️ Protect",
    "Utility": "📊 Utility",
    "Compliance": "✅ Compliance",
    "Report": "📝 Report"
}
selected = st.sidebar.radio("Go to", list(steps.keys()), format_func=lambda x: steps[x], index=0)

# ------------------ SESSION STATE ------------------
if "df_real" not in st.session_state: st.session_state.df_real = None
if "df_anon" not in st.session_state: st.session_state.df_anon = None
if "df_protected" not in st.session_state: st.session_state.df_protected = None
if "risk" not in st.session_state: st.session_state.risk = {}
if "checklist" not in st.session_state: st.session_state.checklist = comp.default_checklist()
if "config" not in st.session_state: st.session_state.config = {}

# ------------------ HELPER ------------------
def load_csv(uploaded):
    if uploaded is None:
        return None
    return pd.read_csv(uploaded)

# ------------------ CONFIG SAVE/LOAD ------------------
with st.sidebar.expander("⚙️ Save/Load Config"):
    cfg = st.session_state.config
    if st.button("💾 Save current config"):
        cfg_bytes = io.BytesIO(yaml.safe_dump(cfg).encode("utf-8"))
        st.download_button("⬇️ Download config.yaml", data=cfg_bytes.getvalue(), file_name="config.yaml", mime="text/yaml")
    cfg_file = st.file_uploader("Upload config.yaml", type=["yaml", "yml"], key="cfg_upl")
    if cfg_file is not None:
        st.session_state.config = yaml.safe_load(cfg_file.read()) or {}
        st.success("✅ Config loaded.")

# ------------------ UPLOAD ------------------
if selected == "Upload":
    st.subheader("1️⃣ Upload Datasets")
    c1, c2 = st.columns(2)

    with c1:
        real_file = st.file_uploader("📂 Real-ID dataset (for linkage attack simulation)", type=["csv"])
        if real_file:
            st.session_state.df_real = load_csv(real_file)
            st.dataframe(st.session_state.df_real.head())

    with c2:
        anon_file = st.file_uploader("📂 Anonymous dataset (to be shared)", type=["csv"])
        if anon_file:
            st.session_state.df_anon = load_csv(anon_file)
            st.dataframe(st.session_state.df_anon.head())

    st.info("💡 Tip: Include quasi-identifiers like *Age, Gender, Pincode, City...*")

# ------------------ RISK ------------------
if selected == "Risk":
    st.subheader("2️⃣ Risk Assessment")
    if st.session_state.df_real is None or st.session_state.df_anon is None:
        st.warning("⚠️ Please upload both datasets first.")
    else:
        df_real = st.session_state.df_real
        df_anon = st.session_state.df_anon

        st.write("💡 Quasi-ID suggestions:", risk_mod.QUASI_ID_SUGGESTIONS)
        quasi = st.multiselect("Select quasi-identifiers", options=df_anon.columns.tolist(),
                               default=[c for c in risk_mod.QUASI_ID_SUGGESTIONS if c in df_anon.columns])

        try:
            with st.spinner("🔎 Running risk assessment..."):
                risk_score, row_scores, nn_idx, dists = risk_mod.simulate_reidentification_risk(df_anon, df_real, quasi)
            st.metric("Estimated Linkage Risk (0-1)", f"{risk_score:.3f}")
            st.session_state.risk = {"risk_score": risk_score, "quasi": quasi}
            st.line_chart(pd.Series(row_scores, name="Row risk"))
            st.success("✅ Risk assessment completed!")
        except Exception as e:
            st.error(f"❌ Risk estimation failed: {e}")

# ------------------ PROTECT ------------------
if selected == "Protect":
    st.subheader("3️⃣ Privacy Enhancement")
    if st.session_state.df_anon is None:
        st.warning("⚠️ Upload an anonymous dataset first.")
    else:
        df = st.session_state.df_anon
        sugg = priv.smart_suggest(df)
        st.write("💡 Smart suggestions:", sugg)

        sdc_cols = st.multiselect("🔒 Suppress categorical cols", options=df.columns.tolist(), default=sugg["sdc_cols"][:5])
        gen_cols = st.multiselect("📦 Generalize numeric cols (binning)", options=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])], default=sugg["generalize_cols"][:5])
        dp_cols = st.multiselect("➕ Add DP noise to numeric cols", options=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])], default=sugg["dp_cols"][:5])
        epsilon = st.slider("🔑 DP epsilon (lower = more privacy)", min_value=0.1, max_value=5.0, step=0.1, value=float(sugg["dp_epsilon"]))
        method = st.selectbox("Or choose Synthetic Data generation", ["None", "Lightweight sampler"])

        df_prot = df.copy()
        if "income" in df_prot.columns:
            df_prot['income_binned'] = pd.cut(df_prot['income'], bins=5)

        df_prot = priv.sdc_suppress(df_prot, sdc_cols, threshold=5)
        df_prot = priv.generalize_numeric(df_prot, gen_cols, bins=10)
        dp_cols = [c for c in dp_cols if pd.api.types.is_numeric_dtype(df_prot[c])]
        df_prot = priv.add_dp_noise(df_prot, dp_cols, epsilon=epsilon, sensitivity=1.0)

        if method == "Lightweight sampler":
            df_prot = priv.synthetic_sample(df_prot, n=len(df))

        st.session_state.df_protected = df_prot
        st.success("✅ Protection applied.")
        st.dataframe(df_prot.head())
        st.download_button("⬇️ Download protected.csv", df_prot.to_csv(index=False), file_name="protected.csv", mime="text/csv")

# ------------------ UTILITY ------------------
if selected == "Utility":
    st.subheader("4️⃣ Utility Measurement")
    if st.session_state.df_anon is None or st.session_state.df_protected is None:
        st.warning("⚠️ Run protection first.")
    else:
        before = st.session_state.df_anon
        after = st.session_state.df_protected

        tab1, tab2, tab3 = st.tabs(["📊 Stats", "📉 Drift", "🤖 ML Utility"])
        with tab1:
            st.write("Overall stats BEFORE")
            st.dataframe(util.basic_stats(before))
            st.write("Overall stats AFTER")
            st.dataframe(util.basic_stats(after))

        with tab2:
            drift = util.distribution_drift(before, after)
            st.write("Distribution drift")
            st.dataframe(drift)

        with tab3:
            target = st.selectbox("Select target column for ML utility check", options=["<none>"] + before.columns.tolist(), index=0)
            if target != "<none>":
                util_results = util.model_utility_check(before, after, target)
                st.write("Model utility (LogReg/RF fallback)")
                st.dataframe(util_results)

# ------------------ COMPLIANCE ------------------
if selected == "Compliance":
    st.subheader("5️⃣ Compliance Checklist (DPDP/GDPR)")
    df = st.session_state.checklist
    edit = st.data_editor(df, num_rows="dynamic", key="chkedit")
    st.session_state.checklist = edit
    score = comp.score(edit)
    st.metric("Checklist completion", f"{int(score*100)}%")

# ------------------ REPORT ------------------
if selected == "Report":
    st.subheader("6️⃣ Reporting & Export")

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

    html_path = os.path.abspath("safedata_report.html")
    pdf_path = os.path.abspath("safedata_report.pdf")
    comp_df = st.session_state.checklist
    risk_summary = st.session_state.get("risk", {})

    rep.save_html_report(html_path, summary, risk_summary, util_stats, comp_df)
    pdf_file = rep.try_make_pdf(html_path, pdf_path)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows Before", summary["rows_before"])
    col2.metric("Rows After", summary["rows_after"])
    col3.metric("Risk Score", f"{summary['risk_score']:.3f}" if summary["risk_score"] else "N/A")

    with open(html_path, "rb") as f:
        st.download_button("⬇️ Download Report (HTML)", data=f.read(), file_name="safedata_report.html", mime="text/html")
    if pdf_file and os.path.exists(pdf_file):
        with open(pdf_file, "rb") as f:
            st.download_button("⬇️ Download Report (PDF)", data=f.read(), file_name="safedata_report.pdf", mime="application/pdf")
    else:
        st.info("ℹ️ PDF export uses a simple fallback. Install 'reportlab' for better support.")
