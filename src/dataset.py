import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from config import DATA_PATH, RANDOM_SEED
from utils import set_seed

set_seed()

def load_hancock():
    df = pd.read_csv(DATA_PATH)
    # exclusion rule：lack Histopathology / Surgery report / Blood
    mask = (~df["histopathology"].isna()) & (~df["surgery_report"].isna()) & (~df["blood"].isna())
    df = df.loc[mask].copy()
    print(f"After exclusion sample size {df.shape[0]}")
    return df

def encode_icd_bag_of_words(df):
    """surgery report ICD‑codes bag‑of‑words, keep codes >=3 patients"""
    corpus = df["icd_codes"].fillna("").astype(str).tolist()
    vec = CountVectorizer(min_df=3, token_pattern=r"[^, ]+")
    icd_mat = vec.fit_transform(corpus).toarray()
    icd_df = pd.DataFrame(icd_mat, columns=[f"icd_{c}" for c in vec.get_feature_names_out()])
    df = pd.concat([df.reset_index(drop=True), icd_df.reset_index(drop=True)], axis=1)
    return df, vec

def preprocess_features(df):
    """encoding categorical / ordinal / numerical, return feature matrix and survival label"""
    # ordinal pT pN
    df["pT_ord"] = df["pT_stage"].map({"T1":1,"T2":2,"T3":3,"T4":4})
    df["pN_ord"] = df["pN_stage"].map({"N0":0,"N1":1,"N2":2,"N3":3,"Nx":np.nan})

    cat_cols = ["sex","smoking_status","primary_site","histologic_type","resection_status"]
    num_cols = ["age","granulocytes","lymphocytes","hemoglobin","platelets","rdw",
                "tma_cd3_inv_density","tma_cd8_inv_density"]

    ohe = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
    cat_arr = ohe.fit_transform(df[cat_cols].fillna("unknown"))
    cat_df = pd.DataFrame(cat_arr, columns=ohe.get_feature_names_out(cat_cols))

    num_df = df[num_cols].copy()
    fea_df = pd.concat([cat_df.reset_index(drop=True), num_df.reset_index(drop=True)], axis=1)

    # survival label for pycox / sksurv
    surv_os = pd.DataFrame()
    surv_os["duration"] = df["os_months"]
    surv_os["event"] = df["os_deceased"].astype(int)

    surv_rfs = pd.DataFrame()
    surv_rfs["duration"] = df["rfs_months"]
    surv_rfs["event"] = df["rfs_recurrence"].astype(int)

    return fea_df, surv_os, surv_rfs, ohe

def mice_impute_fold(train_fea, test_fea):
    """MICE imputation strictly fit only on train fold avoid leakage"""
    imputer = IterativeImputer(random_state=RANDOM_SEED, max_iter=20)
    train_imp = imputer.fit_transform(train_fea)
    test_imp = imputer.transform(test_fea)
    return train_imp, test_imp
