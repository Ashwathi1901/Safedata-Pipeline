import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# put this in modules/utility.py (replace old basic_stats)
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
from scipy import stats
# other imports (keep your ks/chi2/model functions unchanged)

def basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust column-by-column stats safe for Streamlit display.
    Converts non-numeric/interval/categorical values to strings for label summaries.
    """
    if df is None:
        return pd.DataFrame([{"column": "(no data)"}])
    if df.empty:
        return pd.DataFrame([{"column": "(empty dataframe)"}])

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
            "count": int(n),
            "missing": miss,
            "missing_%": miss_pct,
            "nunique": nunique,
        }

        # numeric columns
        if is_numeric_dtype(s):
            s_num = pd.to_numeric(s, errors="coerce").dropna()
            row.update({
                "mean": float(s_num.mean()) if not s_num.empty else np.nan,
                "std": float(s_num.std()) if not s_num.empty else np.nan,
                "min": float(s_num.min()) if not s_num.empty else np.nan,
                "p25": float(s_num.quantile(0.25)) if not s_num.empty else np.nan,
                "median": float(s_num.median()) if not s_num.empty else np.nan,
                "p75": float(s_num.quantile(0.75)) if not s_num.empty else np.nan,
                "max": float(s_num.max()) if not s_num.empty else np.nan,
            })
        # datetime columns
        elif is_datetime64_any_dtype(s):
            s_dt = pd.to_datetime(s, errors="coerce")
            try:
                row.update({
                    "min": str(s_dt.min()),
                    "max": str(s_dt.max()),
                })
            except Exception:
                row.update({"min": None, "max": None})
        # everything else: safe label summary
        else:
            s_safe = s.astype(str)  # safe conversion for intervals/categories/etc.
            try:
                top_val = s_safe.mode(dropna=True).iloc[0]
                top_freq = int(s_safe.value_counts(dropna=True).iloc[0])
            except Exception:
                top_val, top_freq = None, np.nan
            row.update({
                "top": None if top_val is None else str(top_val),
                "freq": top_freq
            })

        rows.append(row)

    # build DataFrame and keep consistent column order
    order = ["column","dtype","count","missing","missing_%","nunique",
             "mean","std","min","p25","median","p75","max","top","freq"]
    out = pd.DataFrame(rows)
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
