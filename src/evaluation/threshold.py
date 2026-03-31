import numpy as np

def compute_threshold(mse_values):
    threshold = np.percentile(mse_values, 99)
    return threshold