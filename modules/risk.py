import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors

QUASI_ID_SUGGESTIONS = ["age", "gender", "zipcode", "pincode", "city", "state", "education", "income"]

def build_linkage_pipeline(df: pd.DataFrame, quasi_ids):
    quasi_ids = [c for c in quasi_ids if c in df.columns]
    cat_cols = [c for c in quasi_ids if df[c].dtype == "object"]
    num_cols = [c for c in quasi_ids if c not in cat_cols]
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(with_mean=True, with_std=True), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
    if not transformers:
        raise ValueError("No valid quasi-identifiers found to assess risk.")
    pre = ColumnTransformer(transformers)
    pipe = Pipeline([("pre", pre)])
    X = pipe.fit_transform(df)
    return pipe, X

def simulate_reidentification_risk(df_anon: pd.DataFrame, df_real: pd.DataFrame, quasi_ids, k=1):
    pipe, X_anon = build_linkage_pipeline(df_anon, quasi_ids)
    X_real = pipe.transform(df_real[quasi_ids])
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(X_real)
    distances, indices = knn.kneighbors(X_anon, n_neighbors=k, return_distance=True)
    row_max = distances.max(axis=1) + 1e-9
    row_scores = 1 - (distances[:, 0] / (row_max))
    overall_risk = float(np.clip(row_scores.mean(), 0, 1))
    return overall_risk, row_scores.tolist(), indices[:, 0].tolist(), distances[:, 0].tolist()
