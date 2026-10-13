import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from config import TIME_HORIZON, OUTPUT_DIR

def calc_cindex(model, X, y_surv):
    risk = model.predict(X)
    cidx, _, _, _, _ = concordance_index_censored(y_surv["event"].astype(bool), y_surv["duration"], risk)
    return cidx

def plot_km_curve(risk_score, y_surv, save_name="km_plot.png"):
    median_risk = np.median(risk_score)
    high_mask = risk_score >= median_risk
    low_mask = risk_score < median_risk

    km_high = KaplanMeierFitter()
    km_low = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(8,6))
    km_high.fit(durations=y_surv.loc[high_mask,"duration"], event_observed=y_surv.loc[high_mask,"event"], label="High risk")
    km_low.fit(durations=y_surv.loc[low_mask,"duration"], event_observed=y_surv.loc[low_mask,"event"], label="Low risk")
    km_high.plot(ax=ax)
    km_low.plot(ax=ax)
    res = logrank_test(y_surv.loc[high_mask,"duration"], y_surv.loc[low_mask,"duration"],
                       event_observed_A=y_surv.loc[high_mask,"event"],
                       event_observed_B=y_surv.loc[low_mask,"event"])
    ax.set_title(f"Kaplan‑Meier, log‑rank p={res.p_value:.4f}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{save_name}", dpi=200)
    plt.close()
    return res.p_value
