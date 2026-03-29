import numpy as np

def compute_threshold(mse_values):
    """
    Computes anomaly threshold using IQR rule.
        threshold = Q3 + 1.5 * (Q3 - Q1)
    """

    Q1 = np.percentile(mse_values, 25)
    Q3 = np.percentile(mse_values, 75)

    iqr = Q3 - Q1
    threshold = Q3 + 1.5 * iqr

    return threshold