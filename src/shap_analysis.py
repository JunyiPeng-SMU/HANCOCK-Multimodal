import shap
import matplotlib.pyplot as plt
import pandas as pd
from config import OUTPUT_DIR

def shap_beeswarm_plot(model_gbdt, X_test_df, save_fig="shap_beeswarm.png"):
    explainer = shap.TreeExplainer(model_gbdt)
    shap_values = explainer.shap_values(X_test_df)
    plt.figure()
    shap.summary_plot(shap_values, X_test_df, show=False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{save_fig}", dpi=200, bbox_inches="tight")
    plt.close()
    return shap_values
