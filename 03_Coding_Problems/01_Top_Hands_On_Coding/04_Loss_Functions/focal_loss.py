import numpy as np

def focal_loss_v1(y_true, y_pred_prob, gamma=2.0):
    """
    y_true: one-hot 编码, (N,C)
    y_pred_prob: 预测概率(N,C)
    theta: 次幂, scalar
    """
    
    epsilon = 1e-15
    y_pred_prob = np.clip(y_pred_prob, epsilon, 1 - epsilon)
    
    pt =  np.sum(y_true * y_pred_prob, axis=1) # (N,1)
    loss = -np.power(1 - pt, gamma) * np.log(pt)

    return np.mean(loss)



def softmax(logits):
    """
    logits: (N,C)
    """

    logits_mx = np.max(logits, axis=1, keepdims=True) # (N,1)
    logits_scale = logits - logits_mx # (N,C)
    

    logits_exp = np.exp(logits_scale) # (N,C)
    logits_exp_sum = np.sum(logits_exp, axis=1, keepdims=True) # (N,1)
    logits_exp = logits_exp / logits_exp_sum # (N,C)
    return logits_exp


def focal_loss_v2(inputs, targets, gamma=2.0, alpha=None, epsilon=1e-15):
    """
    inputs: (N,C)  raw logits of model. N is batch_size, C is class_num
    targets: (N,) 每个元素是类别索引 Integer values from 0 to C-1.
    """
    N, C = inputs.shape
    probs = softmax(inputs) #(N,C)
    
    # 传入多个数组（或列表等序列）来进行所谓的“高级索引” ,NumPy 会将它们按元素配对来确定要选择的元素坐标
    pt = probs[np.arange(N), targets] # (N,)
    
    pt = pt + epsilon
    
    loss = -1 * np.power(1 - pt, gamma) * np.log(pt)
    
    return np.mean(loss)
    
    
    
    
    
    
    