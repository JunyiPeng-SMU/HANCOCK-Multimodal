import random
import numpy as np
import pandas as pd
import pickle
from config import RANDOM_SEED

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)

def save_pkl(obj, path):
    with open(path,"wb") as f:
        pickle.dump(obj,f)

def load_pkl(path):
    with open(path,"rb") as f:
        return pickle.load(f)

def save_metric_csv(df, filename):
    out_path = os.path.join(config.OUTPUT_DIR, filename)
    df.to_csv(out_path, index=False)
