"""Main run script"""
import os
import pandas as pd
from src.config import *
from src.utils import set_seed
from src.dataset import load_hancock, encode_icd_bag_of_words, preprocess_features, mice_impute_fold
from src.feature_selection import remove_high_spearman_corr, univariate_cox_screen, lasso_cox_selection
from src.models import get_model
from src.evaluation import calc_cindex, plot_km_curve
from src.shap_analysis import shap_beeswarm_plot

set_seed(RANDOM_SEED)

def main():
    print("==== Load HANCOCK dataset ====")
    df_raw = load_hancock()
    df_raw, icd_vectorizer = encode_icd_bag_of_words(df_raw)
    fea_df, surv_os, surv_rfs, ohe_encoder = preprocess_features(df_raw)

    print("==== Feature selection pipeline ====")
    fea_spear, cols_spear = remove_high_spearman_corr(fea_df)
    fea_univar, cols_univar = univariate_cox_screen(fea_spear, surv_os)
    fea_lasso, cols_lasso = lasso_cox_selection(fea_univar, surv_os, LASSO_ALPHA_RANGE)
    print(f"Final selected feature count: {len(cols_lasso)}")

    # ========= note =========
    # In original paper train‑test split uses genetic‑algorithm;
    # In this demo replace with random split, you can replace with original split index from HANCOCK supplement
    from sklearn.model_selection import train_test_split
    idx_train, idx_test = train_test_split(fea_lasso.index, train_size=TRAIN_TEST_RATIO, random_state=RANDOM_SEED)
    X_train_df = fea_lasso.loc[idx_train].copy()
    X_test_df = fea_lasso.loc[idx_test].copy()
    y_train_os = surv_os.loc[idx_train].copy()
    y_test_os = surv_os.loc[idx_test].copy()

    X_train_imp, X_test_imp = mice_impute_fold(X_train_df, X_test_df)
    X_train_df = pd.DataFrame(X_train_imp, columns=X_train_df.columns)
    X_test_df = pd.DataFrame(X_test_imp, columns=X_test_df.columns)

    model_list = ["coxph","survival_svm","rsf","gbdt_survival"]
    metric_records = []
    for name in model_list:
        print(f"\nTrain model {name}")
        model = get_model(name, random_state=RANDOM_SEED)
        model.fit(X_train_df.values, y_train_os)
        cindex_test = calc_cindex(model, X_test_df.values, y_test_os)
        risk_test = model.predict(X_test_df.values)
        plot_km_curve(risk_test, y_test_os, save_name=f"km_{name}_os.png")
        metric_records.append({"model":name,"cindex_test_os":cindex_test})

    metric_df = pd.DataFrame(metric_records)
    metric_df.to_csv(os.path.join(OUTPUT_DIR,"model_metrics.csv"), index=False)

    # SHAP for best model GBDT‑Survival
    print("\nRun SHAP analysis for GBDT‑Survival")
    gbdt_model = get_model("gbdt_survival")
    gbdt_model.fit(X_train_df.values, y_train_os)
    shap_vals = shap_beeswarm_plot(gbdt_model, X_test_df)

    print(f"All output saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
