import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sksurv.linear_model import CoxPHSurvivalAnalysis
from pycox.models import CoxPH
from sklearn.linear_model import LassoCV
from config import SPEARMAN_THRESHOLD, UNIVAR_COX_P_THRESHOLD

def remove_high_spearman_corr(X_df, threshold=SPEARMAN_THRESHOLD):
    corr_mat = X_df.corr(method="spearman").abs()
    upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > threshold)]
    keep_cols = [c for c in X_df.columns if c not in drop_cols]
    return X_df[keep_cols], keep_cols

def univariate_cox_screen(X_df, y_surv, p_cut=UNIVAR_COX_P_THRESHOLD):
    """univariable Cox‑PH screen, keep p < threshold"""
    keep = []
    for col in X_df.columns:
        X_col = X_df[[col]].copy()
        cox = CoxPHSurvivalAnalysis()
        cox.fit(X_col, y_surv)
        p_val = cox.summary_.p_values[0]
        if p_val < p_cut:
            keep.append(col)
    return X_df[keep], keep

def lasso_cox_selection(X_train, y_train_surv, alpha_list):
    """LASSO‑Cox penalized feature selection using pycox CoxPH LassoCV"""
    lasso_cv = LassoCV(alphas=alpha_list, cv=5, random_state=42)
    lasso_cv.fit(X_train, y_train_surv["duration"])
    mask = np.abs(lasso_cv.coef_) > 1e‑6
    selected_cols = X_train.columns[mask].tolist()
    return X_train[selected_cols], selected_cols
