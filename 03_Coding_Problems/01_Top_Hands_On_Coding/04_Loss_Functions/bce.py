import numpy as np


def binary_cross_entropy(y_true, y_pred_prob):
    """
    计算二元交叉熵损失
    
    y_true: 真值类别, 0或者1, (N,)
    y_pred_prob: 预测为1的概率, (N,)
    """
    # 为了防止log(0)导致数值问题, 对y_pred_prob数组进行裁剪
    epsilon = 1e-15
    # 这里记得赋值。默认np.clip不会改变原数组
    y_pred_prob = np.clip(y_pred_prob, epsilon, 1 - epsilon)
    
    loss = -(y_true * np.log(y_pred_prob) + (1 - y_true) * np.log(1 - y_pred_prob))
    return np.mean(loss)


if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 0])
    y_pred_prob  = np.array([0.9, 0.2, 0.8, 0.1])
    bce_loss = binary_cross_entropy(y_true, y_pred_prob)
    print(bce_loss)
    