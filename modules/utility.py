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

    try:
        # Try all columns
        desc = df.describe(include="all").T
    except Exception as e:
        # Fallback: only numeric summary
        desc = df.describe(include=[np.number]).T
        desc["note"] = f"Non-numeric columns skipped due to error: {str(e)}"

    # Always add missing counts
    desc["missing"] = df.isnull().sum()

    return desc


    try:
        desc = df.describe(include="all").T
    except Exception:
        # Fallback to numeric-only summary if mixed data causes errors
        desc = df.describe().T

    # Add missing values (count instead of fraction for clarity)
    desc["missing"] = df.isnull().sum()
    return desc


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
