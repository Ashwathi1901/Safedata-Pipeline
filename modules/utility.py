import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def basic_stats(df: pd.DataFrame):
    if df is None or df.empty:
        return pd.DataFrame({"message": ["No data available"]})

    results = []

    # Numeric stats
    num_cols = df.select_dtypes(include=[np.number])
    if not num_cols.empty:
        num_desc = num_cols.describe().T
        num_desc["missing"] = num_cols.isnull().sum()
        results.append(num_desc)

    # Object stats (gender, name, etc.)
    obj_cols = df.select_dtypes(include=["object"])
    if not obj_cols.empty:
        obj_desc = obj_cols.describe().T
        obj_desc["missing"] = obj_cols.isnull().sum()
        results.append(obj_desc)

    # Category stats (like pd.cut intervals)
    cat_cols = df.select_dtypes(include=["category"])
    if not cat_cols.empty:
        # 🔑 Convert categories (including pd.cut intervals) to string
        cat_as_str = cat_cols.astype(str)
        cat_desc = cat_as_str.describe().T
        cat_desc["missing"] = cat_as_str.isnull().sum()
        results.append(cat_desc)

    if results:
        return pd.concat(results, axis=0)
    else:
        return pd.DataFrame({"message": ["No valid columns to summarize"]})



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
