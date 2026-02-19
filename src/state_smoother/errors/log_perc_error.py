import numpy as np
import torch as pt

def log_perc_error(prediction, actual):
    """ Calculates the log of percentage error
    
    Calculates the log of the percentage of error:
    $E = ln((P - A) / A)$
    With P = prediction, and A = actual.
    Clamping is needed on the predictions to ensure values do not become
    negative, and again on the pre-logged result to ensure values do not
    overflow the log calculation.
    
    Parameters
    ----------
    prediction: PyTorch Tensor
        Predicted state to calculate error with.
        Shape: Dims, Columns
    actual: PyTorch Tensor
        Actual state to calculate error against.
        Shape: Dims, Columns
        
    Returns
    -------
    error: PyTorch Tensor
        Log of percentage error.
    """
    
    error = (prediction.clamp(0, np.inf) - actual) / actual
    error = pt.log(error.clamp(1e-16, np.inf))
    return error