import numpy as np
import pandas as pd

def sdc_suppress(df: pd.DataFrame, cols, threshold=5):
    df2 = df.copy()
    for col in cols:
        if col in df2.columns and df2[col].dtype == "object":
            vc = df2[col].value_counts(dropna=False)
            rare = vc[vc < threshold].index
            df2[col] = df2[col].where(~df2[col].isin(rare), "OTHER")
    return df2

def generalize_numeric(df: pd.DataFrame, cols, bins=10):
    df2 = df.copy()
    for col in cols:
        if col in df2.columns and np.issubdtype(df2[col].dtype, np.number):
            real_bins = pd.qcut(df2[col], q=bins, retbins=True, labels=False, duplicates="drop")[1]
            labels = []
            for i in range(len(real_bins)-1):
                labels.append(f"[{real_bins[i]:.2f}, {real_bins[i+1]:.2f})")
            df2[col] = pd.qcut(df2[col], q=bins, labels=labels, duplicates="drop")
    return df2

def add_dp_noise(df: pd.DataFrame, cols, epsilon=1.0, sensitivity=1.0):
    df2 = df.copy()
    scale = sensitivity / max(epsilon, 1e-6)
    for col in cols:
        if col in df2.columns and np.issubdtype(df2[col].dtype, np.number):
            noise = np.random.laplace(loc=0.0, scale=scale, size=len(df2))
            df2[col] = df2[col] + noise
    return df2

def synthetic_sample(df: pd.DataFrame, n=None, seed=42):
    rng = np.random.default_rng(seed)
    if n is None:
        n = len(df)
    synth = {}
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            synth[col] = pd.Series([np.nan]*n)
            continue
        if np.issubdtype(s.dtype, np.number):
            mu, sigma = s.mean(), s.std(ddof=1) or 1.0
            m = int(n*0.5)
            boot = s.sample(n=n-m, replace=True, random_state=seed).to_numpy()
            gauss = rng.normal(mu, sigma, size=m)
            synth[col] = pd.Series(np.concatenate([boot, gauss]))[:n]
        else:
            vals = s.value_counts(normalize=True)
            choices = rng.choice(vals.index.to_list(), size=n, p=vals.values)
            synth[col] = pd.Series(choices)
    return pd.DataFrame(synth)

def smart_suggest(df: pd.DataFrame):
    suggestions = {"sdc_cols": [], "generalize_cols": [], "dp_cols": [], "dp_epsilon": 1.0}
    for col in df.columns:
        unique = df[col].nunique(dropna=True)
        if df[col].dtype == "object":
            if unique > 20:
                suggestions["sdc_cols"].append(col)
        else:
            if unique > 50:
                suggestions["generalize_cols"].append(col)
                suggestions["dp_cols"].append(col)
            else:
                suggestions["dp_cols"].append(col)
    return suggestions
