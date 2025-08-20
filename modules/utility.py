import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust summary that never calls pandas.describe() (avoids CategoricalDtype / Interval issues).
    Returns only simple Python/NumPy scalars/strings that Streamlit can render safely.
    """
    if df is None or df.empty:
        return pd.DataFrame([{"column": "(no data)"}])

    rows = []
    for col in df.columns:
        s = df[col]
        n = len(s)
        miss = int(s.isna().sum())
        miss_pct = (miss / n * 100.0) if n else np.nan
        nunique = int(s.nunique(dropna=True))

        row = {
            "column": col,
            "dtype": str(s.dtype),
            "count": n,
            "missing": miss,
            "missing_%": round(miss_pct, 2) if n else np.nan,
            "nunique": nunique,
        }

        if is_numeric_dtype(s):
            s_num = pd.to_numeric(s, errors="coerce")
            row.update({
                "mean": float(s_num.mean()) if n else np.nan,
                "std": float(s_num.std()) if n else np.nan,
                "min": float(s_num.min()) if n else np.nan,
                "p25": float(s_num.quantile(0.25)) if n else np.nan,
                "median": float(s_num.median()) if n else np.nan,
                "p75": float(s_num.quantile(0.75)) if n else np.nan,
                "max": float(s_num.max()) if n else np.nan,
            })
        elif is_datetime64_any_dtype(s):
            # Show range for datetimes
            row.update({
                "min": str(pd.to_datetime(s, errors="coerce").min()),
                "max": str(pd.to_datetime(s, errors="coerce").max()),
            })
        else:
            # Objects / categories / intervals / bools → treat as labels
            vc = s.astype("string")  # string dtype is display-safe
            try:
                top_val = vc.mode(dropna=True).iloc[0]
                top_freq = vc.value_counts(dropna=True).iloc[0]
            except Exception:
                top_val, top_freq = pd.NA, pd.NA
            row.update({
                "top": None if pd.isna(top_val) else str(top_val),
                "freq": int(top_freq) if pd.notna(top_freq) else np.nan,
            })

        rows.append(row)

    order = ["column","dtype","count","missing","missing_%","nunique",
             "mean","std","min","p25","median","p75","max","top","freq"]
    out = pd.DataFrame(rows)
    # Keep only columns that exist in this run
    out = out[[c for c in order if c in out.columns]]
    return out



def ks_numeric(a: pd.Series, b: pd.Series):
    a = a.dropna()
    b = b.dropna()
    if len(a) < 5 or len(b) < 5:
        return np.nan
    return stats.ks_2samp(a, b).statistic

def chi2_categorical(a: pd.Series, b: pd.Series):
    a = a.fillna("NA").astype(str)
    b = b.fillna("NA").astype(str)
    ta = a.value_counts()
    tb = b.value_counts()
    cats = sorted(set(ta.index).union(tb.index))
    oa = np.array([ta.get(c, 0) for c in cats])
    ob = np.array([tb.get(c, 0) for c in cats])
    if oa.sum()==0 or ob.sum()==0:
        return np.nan
    chi2 = ((oa - ob)**2 / (oa + ob + 1e-9)).sum()
    return chi2

def distribution_drift(df_before: pd.DataFrame, df_after: pd.DataFrame):
    metrics = []
    for col in df_before.columns:
        if col not in df_after.columns:
            continue
        if np.issubdtype(df_before[col].dtype, np.number) and np.issubdtype(df_after[col].dtype, np.number):
            m = ks_numeric(df_before[col], df_after[col])
            metrics.append({"column": col, "type": "numeric", "ks_stat": m})
        else:
            m = chi2_categorical(df_before[col], df_after[col])
            metrics.append({"column": col, "type": "categorical", "chi2": m})
    return pd.DataFrame(metrics)

def model_utility_check(df_before: pd.DataFrame, df_after: pd.DataFrame, target: str):
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y if y.nunique()<20 else None)
        clf = LogisticRegression(max_iter=200)
        try:
            clf.fit(X_train, y_train)
        except Exception:
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        results.append({"dataset": name, "acc": accuracy_score(y_test, y_pred), "f1": f1_score(y_test, y_pred, average="weighted")})
    return pd.DataFrame(results)
