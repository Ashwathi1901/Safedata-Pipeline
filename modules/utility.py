import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ─────────────────────────────────────────────
# BASIC STATS
# ─────────────────────────────────────────────
def basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute robust column-wise summary statistics.
    Safe for Streamlit display (handles empty / mixed dtypes).
    """
    if df is None or df.empty:
        return pd.DataFrame([{"column": "(no data)"}])

    rows = []
    for col in df.columns:
        s = df[col]
        n = len(s)
        miss = int(s.isna().sum())
        miss_pct = round(miss / n * 100.0, 2) if n else np.nan
        nunique = int(s.nunique(dropna=True))

        row = {
            "column": col,
            "dtype": str(s.dtype),
            "count": n,
            "missing": miss,
            "missing_%": miss_pct,
            "nunique": nunique,
        }

        if is_numeric_dtype(s):
            s_num = pd.to_numeric(s, errors="coerce").dropna()
            if not s_num.empty:
                row.update({
                    "mean": float(s_num.mean()),
                    "std": float(s_num.std()),
                    "min": float(s_num.min()),
                    "p25": float(s_num.quantile(0.25)),
                    "median": float(s_num.median()),
                    "p75": float(s_num.quantile(0.75)),
                    "max": float(s_num.max()),
                })
        elif is_datetime64_any_dtype(s):
            s_dt = pd.to_datetime(s, errors="coerce").dropna()
            if not s_dt.empty:
                row.update({
                    "min": str(s_dt.min()),
                    "max": str(s_dt.max()),
                })
        else:
            s_safe = s.astype(str).fillna("NA")
            if not s_safe.empty:
                top_val = s_safe.mode().iloc[0] if not s_safe.mode().empty else None
                top_freq = int(s_safe.value_counts().iloc[0]) if not s_safe.value_counts().empty else 0
                row.update({"top": top_val, "freq": top_freq})

        rows.append(row)

    order = ["column","dtype","count","missing","missing_%","nunique",
             "mean","std","min","p25","median","p75","max","top","freq"]
    out = pd.DataFrame(rows)
    return out[[c for c in order if c in out.columns]]


# ─────────────────────────────────────────────
# DISTRIBUTION DRIFT
# ─────────────────────────────────────────────
def ks_numeric(a: pd.Series, b: pd.Series):
    a, b = a.dropna(), b.dropna()
    if len(a) < 5 or len(b) < 5:
        return np.nan
    return stats.ks_2samp(a, b).statistic

def chi2_categorical(a: pd.Series, b: pd.Series):
    a = a.fillna("NA").astype(str)
    b = b.fillna("NA").astype(str)
    cats = sorted(set(a.unique()).union(set(b.unique())))
    oa = np.array([(a == c).sum() for c in cats])
    ob = np.array([(b == c).sum() for c in cats])
    if oa.sum()==0 or ob.sum()==0:
        return np.nan
    # add 1e-9 to avoid zero-division
    chi2, _ = stats.chisquare(f_obs=oa + 1e-9, f_exp=ob + 1e-9)
    return chi2

def distribution_drift(df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:
    metrics = []
    for col in df_before.columns:
        if col not in df_after.columns:
            continue
        s1, s2 = df_before[col], df_after[col]
        if is_numeric_dtype(s1) and is_numeric_dtype(s2):
            m = ks_numeric(s1, s2)
            metrics.append({"column": col, "type": "numeric", "ks_stat": m})
        else:
            m = chi2_categorical(s1, s2)
            metrics.append({"column": col, "type": "categorical", "chi2": m})
    return pd.DataFrame(metrics)


# ─────────────────────────────────────────────
# MODEL UTILITY CHECK
# ─────────────────────────────────────────────
def model_utility_check(df_before: pd.DataFrame, df_after: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Train/test simple ML models on before vs after datasets
    to check predictive utility retention.
    """
    results = []

    for name, d in [("original", df_before), ("protected", df_after)]:
        if target not in d.columns:
            results.append({"dataset": name, "acc": np.nan, "f1": np.nan})
            continue

        X = d.drop(columns=[target]).select_dtypes(include=[np.number]).copy()
        y = d[target]

        if X.shape[1] < 1 or y.nunique() < 2:
            results.append({"dataset": name, "acc": np.nan, "f1": np.nan})
            continue

        X = X.fillna(X.mean(numeric_only=True))
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42,
                stratify=y if y.nunique() < 20 else None
            )
        except Exception:
            results.append({"dataset": name, "acc": np.nan, "f1": np.nan})
            continue

        clf = LogisticRegression(max_iter=200)
        try:
            clf.fit(X_train, y_train)
        except Exception:
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        results.append({
            "dataset": name,
            "acc": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, average="weighted")
        })

    return pd.DataFrame(results)
