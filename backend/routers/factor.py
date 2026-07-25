"""Factor Analysis & Principal Component Analysis (PCA) Router.

Computes KMO, Bartlett's Test of Sphericity, Eigenvalues/Variance,
Loadings Matrices (rotated or unrotated), and Plotly coordinates.
"""

import math
from typing import List, Optional
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from scipy import stats as sp

from services import store
from services.impute import apply_imputation

router = APIRouter()


class FactorPCARequest(BaseModel):
    session_id: str
    items: List[str]
    extraction: str = "pca"              # pca | efa
    rotation: str = "varimax"            # none | varimax | promax
    n_factors: Optional[int] = None       # None means Eigenvalue > 1 (auto)
    imputation: str = "listwise"         # listwise | mean | median


def _safe(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def varimax_rotation(loadings, max_iter=500, tol=1e-6):
    """Orthogonal Varimax rotation using Kaiser normalization."""
    p, k = loadings.shape
    if k < 2:
        return loadings, np.eye(k)
    
    # Kaiser normalization: divide rows by their h2
    h2 = np.sum(loadings**2, axis=1, keepdims=True)
    h2[h2 == 0] = 1e-12
    normalized_loadings = loadings / np.sqrt(h2)
    
    R = np.eye(k)
    d = 0
    for _ in range(max_iter):
        d_old = d
        # Varimax update
        Lambda = np.dot(normalized_loadings, R)
        u, s, vh = np.linalg.svd(
            np.dot(
                normalized_loadings.T,
                Lambda**3 - (1.0 / p) * np.dot(Lambda, np.diag(np.sum(Lambda**2, axis=0)))
            )
        )
        R = np.dot(u, vh)
        d = np.sum(s)
        if d_old != 0 and d - d_old < tol:
            break
            
    # Denormalize rotated loadings
    rotated_loadings = normalized_loadings @ R * np.sqrt(h2)
    return rotated_loadings, R


def promax_rotation(loadings, m=4):
    """Oblique Promax rotation targeting a powered Varimax target."""
    p, k = loadings.shape
    if k < 2:
        return loadings, np.eye(k)
        
    # Promax starts with Varimax pre-rotation
    v_loadings, R_varimax = varimax_rotation(loadings)
    
    # Create target matrix P
    P = np.sign(v_loadings) * np.abs(v_loadings)**m
    
    # Procrustes oblique rotation: regressing target P on loadings
    L_inv = np.linalg.pinv(v_loadings)
    H = L_inv @ P
    
    # Normalize columns of H. np.diag on a 2-D array returns a read-only VIEW,
    # so the guard below raised "assignment destination is read-only" on every
    # call — promax was a guaranteed 500. Copy before writing.
    d = np.diag(H.T @ H).copy()
    d[d == 0] = 1e-12
    Q = H @ np.diag(1.0 / np.sqrt(d))
    
    # Oblique loadings
    rotated_loadings = v_loadings @ Q
    return rotated_loadings, Q


def calculate_kmo(df_items: pd.DataFrame):
    """Calculate Kaiser-Meyer-Olkin (KMO) Measure of Sampling Adequacy."""
    corr_matrix = df_items.corr().values
    p = corr_matrix.shape[1]
    
    # Check if invertible
    try:
        inv_corr = np.linalg.inv(corr_matrix)
    except np.linalg.LinAlgError:
        # Add a tiny ridge if not invertible
        inv_corr = np.linalg.inv(corr_matrix + 1e-6 * np.eye(p))
        
    # Compute partial correlation matrix (anti-image correlation matrix)
    A = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            if i != j:
                denom = np.sqrt(inv_corr[i, i] * inv_corr[j, j])
                A[i, j] = -inv_corr[i, j] / denom if denom > 0 else 0
            else:
                A[i, j] = 1.0
                
    # Sum of squared correlation values (excluding main diagonal)
    sum_r2 = 0.0
    sum_a2 = 0.0
    kmo_per_item = {}
    
    for i, col in enumerate(df_items.columns):
        item_r2 = 0.0
        item_a2 = 0.0
        for j in range(p):
            if i != j:
                r2 = corr_matrix[i, j]**2
                a2 = A[i, j]**2
                sum_r2 += r2
                sum_a2 += a2
                item_r2 += r2
                item_a2 += a2
        # Item-specific KMO
        kmo_per_item[col] = float(item_r2 / (item_r2 + item_a2)) if (item_r2 + item_a2) > 0 else 0.0
        
    overall_kmo = float(sum_r2 / (sum_r2 + sum_a2)) if (sum_r2 + sum_a2) > 0 else 0.0
    return overall_kmo, kmo_per_item


def calculate_bartlett(df_items: pd.DataFrame):
    """Calculate Bartlett's Test of Sphericity."""
    n, p = df_items.shape
    if p < 2:
        return 0.0, 0, 1.0
    corr_matrix = df_items.corr().values
    
    # Determinant of correlation matrix
    try:
        det = np.linalg.det(corr_matrix)
    except np.linalg.LinAlgError:
        det = 0.0
        
    # Cap at a minimum value above zero
    det = max(1e-15, det)
    
    df = int(p * (p - 1) / 2)
    # Bartlett test statistic
    chi2_stat = - (n - 1 - (2 * p + 5) / 6.0) * np.log(det)
    p_val = float(sp.chi2.sf(chi2_stat, df))
    
    return float(chi2_stat), df, p_val


@router.post("/factor_pca")
def factor_pca(req: FactorPCARequest):
    df = store.get_filtered(req.session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")
        
    missing = [c for c in req.items if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Columns not found: {missing}")
        
    if len(req.items) < 3:
        raise HTTPException(status_code=400, detail="Factor Analysis requires at least 3 variables.")
        
    # Slice columns and apply imputation
    df_items = df[req.items].apply(pd.to_numeric, errors="coerce")
    df_items = apply_imputation(df_items, req.items, req.imputation)
    
    # Drop rows that are still missing (in case of listwise)
    df_clean = df_items.dropna()
    n, p = df_clean.shape
    
    if n < 10:
        raise HTTPException(status_code=400, detail=f"Too few observations after imputation (need at least 10, got {n}).")

    # A zero-variance item makes df.corr() emit NaN for its whole row/column,
    # and np.linalg.eigh then dies with "Eigenvalues did not converge" — a 500
    # for what is ordinary user input (e.g. a flag that became constant after
    # filtering). Name the offending columns instead.
    constant_items = [c for c in df_clean.columns if df_clean[c].nunique(dropna=True) <= 1]
    if constant_items:
        raise HTTPException(
            status_code=400,
            detail=(
                f"These items have no variance and cannot be factored: {constant_items}. "
                "Remove them, or widen the filter so they vary."
            ),
        )


    # ── 1. Suitability Tests ──
    overall_kmo, item_kmo = calculate_kmo(df_clean)
    chi2_stat, df_sphericity, p_sphericity = calculate_bartlett(df_clean)
    
    # KMO rating
    if overall_kmo >= 0.9:
        kmo_rating = "Marvelous"
    elif overall_kmo >= 0.8:
        kmo_rating = "Meritorious"
    elif overall_kmo >= 0.7:
        kmo_rating = "Middling"
    elif overall_kmo >= 0.6:
        kmo_rating = "Mediocre"
    elif overall_kmo >= 0.5:
        kmo_rating = "Miserable"
    else:
        kmo_rating = "Unacceptable"
        
    # ── 2. Eigenvalues & Variance Explained (Initial Solution) ──
    # Centered and scaled variables correlation matrix SVD or Eigh
    corr_matrix = df_clean.corr().values
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    
    # eigenvalues returned in ascending order by eigh, reverse them
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    
    total_var = float(p)
    variance_explained = []
    cum_var = 0.0
    
    for idx, lam in enumerate(eigenvalues):
        pct = (lam / total_var) * 100
        cum_var += pct
        variance_explained.append({
            "component": idx + 1,
            "eigenvalue": float(lam),
            "pct_variance": float(pct),
            "cum_variance": float(cum_var)
        })
        
    # Determine number of factors/components
    if req.n_factors is not None and req.n_factors > 0:
        n_fac = min(req.n_factors, p)
    else:
        # Kaiser Criterion: Eigenvalues > 1
        n_fac = max(1, sum(1 for lam in eigenvalues if lam >= 1.0))
        
    # ── 3. Factor/Component Extraction & Loadings ──
    # Initialize unrotated loadings
    if req.extraction == "pca":
        # PCA loadings: eigenvector_j * sqrt(eigenvalue_j)
        unrotated = eigenvectors[:, :n_fac] * np.sqrt(eigenvalues[:n_fac])
        method_label = "Principal Component Analysis (PCA)"
    else:  # Exploratory Factor Analysis (EFA)
        from sklearn.decomposition import FactorAnalysis
        fa = FactorAnalysis(n_components=n_fac, random_state=42, max_iter=1000)
        # Standardise first. KMO, Bartlett and the eigenvalues above all work
        # off the correlation matrix; fitting the factor model on raw units
        # left the loadings on the variables' own scale, so communalities came
        # back above 1 and uniqueness negative — impossible as proportions of
        # variance. z-scoring puts EFA on the same correlation scale as PCA.
        sd = df_clean.std(ddof=0).replace(0, np.nan)
        z = (df_clean - df_clean.mean()) / sd
        fa.fit(z.values)
        # sklearn stores as (n_components, n_features), transpose to (n_features, n_components)
        unrotated = fa.components_.T
        method_label = "Exploratory Factor Analysis (EFA — Principal Axis)"
        
    # Apply rotation
    rot_label = "Unrotated"
    if req.rotation == "varimax":
        rotated, rotation_matrix = varimax_rotation(unrotated)
        rot_label = "Varimax (Orthogonal)"
    elif req.rotation == "promax":
        rotated, rotation_matrix = promax_rotation(unrotated)
        rot_label = "Promax (Oblique)"
    else:
        rotated = unrotated.copy()
        
    # Structure of output loadings
    factors = [f"Factor {i+1}" if req.extraction == "efa" else f"PC{i+1}" for i in range(n_fac)]
    
    loadings_list = []
    # Calculate communality (h2)
    h2 = np.sum(rotated**2, axis=1)
    
    for i, col in enumerate(req.items):
        row_loadings = {
            "variable": col,
            "h2": float(h2[i]),
            "u2": float(1.0 - h2[i])  # uniqueness
        }
        for f_idx, f_name in enumerate(factors):
            row_loadings[f_name] = float(rotated[i, f_idx])
        loadings_list.append(row_loadings)
        
    # Coordinates for Scree Plot
    scree_coords = [{"component": idx + 1, "eigenvalue": float(lam)} for idx, lam in enumerate(eigenvalues)]
    
    # Coordinates for Loadings 2D Biplot (PC1 vs PC2 / Factor 1 vs Factor 2)
    # If 1 component, use PC1 vs index (dummy)
    biplot = []
    for i, col in enumerate(req.items):
        biplot.append({
            "variable": col,
            "x": float(rotated[i, 0]),
            "y": float(rotated[i, 1]) if n_fac > 1 else 0.0
        })
        
    # R replication code
    items_str = ", ".join(f'"{it}"' for it in req.items)
    if req.extraction == "pca":
        r_rot = "none" if req.rotation == "none" else req.rotation
        r_code = (
            f"library(psych)\n"
            f"fit <- principal(data[, c({items_str})], nfactors = {n_fac}, rotate = \"{r_rot}\")\n"
            f"print(fit$loadings, cutoff = 0.3)"
        )
    else:
        r_rot = "none" if req.rotation == "none" else req.rotation
        r_code = (
            f"library(psych)\n"
            f"fit <- fa(data[, c({items_str})], nfactors = {n_fac}, rotate = \"{r_rot}\", fm = \"pa\")\n"
            f"print(fit$loadings, cutoff = 0.3)"
        )
        
    # Return formatted result
    return {
        "test": "Factor & Principal Component Analysis",
        "n": n,
        "p": p,
        "n_factors": n_fac,
        "extraction_method": method_label,
        "rotation_method": rot_label,
        "suitability": {
            "overall_kmo": overall_kmo,
            "kmo_rating": kmo_rating,
            "item_kmo": item_kmo,
            "bartlett_chi2": chi2_stat,
            "bartlett_df": df_sphericity,
            "bartlett_p": p_sphericity
        },
        "variance_explained": variance_explained,
        "loadings": loadings_list,
        "factors": factors,
        "scree_coords": scree_coords,
        "biplot": biplot,
        "r_code": r_code,
        "export_rows": _generate_export_rows(loadings_list, factors)
    }


def _generate_export_rows(loadings, factors):
    """Build CSV/Excel headers and data rows."""
    headers = ["Variable", *factors, "Communality (h2)", "Uniqueness (u2)"]
    rows = [headers]
    for r in loadings:
        row = [
            r["variable"],
            *[r[f] for f in factors],
            r["h2"],
            r["u2"]
        ]
        rows.append(row)
    return rows
