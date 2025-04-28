import numpy as np

def categorical_cross_entropy(y_true, y_pred_prob):
    """
    
    y_true:one-hot编码; shape:(B,C)
    y_pre_prob: shape:(B,C)
    """
    
    eplison = 1e-15
    y_pred_prob = np.clip(y_pred_prob, eplison, 1-eplison)
    
    loss = -1 * np.sum(y_true * np.log(y_pred_prob), axis=-1) # （B,）
    return np.mean(loss)
    
    