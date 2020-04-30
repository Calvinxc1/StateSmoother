import numpy as np
import torch as pt

def log_perc_error(prediction, actual):
    error = pt.log((prediction.clamp(0, np.inf) - actual).clamp(1e-16, np.inf) / actual)
    return error