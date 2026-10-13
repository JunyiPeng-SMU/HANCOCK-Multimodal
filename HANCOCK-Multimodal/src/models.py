from sksurv.ensemble import RandomSurvivalForest
from sksurv.svm import FastSurvivalSVM
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest,GradientBoostingSurvivalAnalysis
from torchmtlr import MTLR
from pycox.evaluation import EvalSurv
import torch
import numpy as np

def get_model(model_name, random_state=42,n_estimators=n_estimators,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,max_features=max_features):
    if model_name == "coxph":
        return CoxPHSurvivalAnalysis()
    elif model_name == "survival_svm":
        return FastSurvivalSVM(max_iter=1000, random_state=random_state)
    elif model_name == "rsf":
        return RandomSurvivalForest(n_estimators=n_estimators, min_samples_split=10, random_state=random_state)
    elif model_name == "neural_mtlr":
        net = torch.nn.Sequential(torch.nn.Linear(104,64), torch.nn.ReLU(), torch.nn.Linear(64,30))
        return MTLR(net,num_time_bins)
    elif model_name == "gbdt_survival":
        return GradientBoostingSurvivalAnalysis(n_estimators=n_estimators,
                               min_samples_split=min_samples_split,
                               min_samples_leaf=min_samples_leaf,
                               max_features=max_features,
                               random_state=random_state)
