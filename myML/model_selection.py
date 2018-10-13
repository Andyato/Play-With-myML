import numpy as np 

def train_test_split(X, y, test_retio=0.2, seed=None):
    assert X.shape[0] == y.shape[0], "样本数和标签数必须一致"
    # assert 0.0 =< test_retio <= 1.0, "切分比例必须在0.0-1.0之间"

    if seed:
        np.random.seed(seed)
    
    shuffled_indexs = np.random.permutation(len(X))
    test = int(len(X) * test_retio)
    test_indexs = shuffled_indexs[:test]
    train_indexs = shuffled_indexs[test:]

    X_train = X[train_indexs]
    y_train = y[train_indexs]

    X_test = X[test_indexs]
    y_test = y[test_indexs]
    
    return X_train, X_test, y_train, y_test

def dJ_debug(J, theta, X_b, y, epsilon=0.01):
    '''
        测试梯度推导是否正确
        J : 损失函数
        return : dJ
    '''
    res = np.empty(len(theta))
    for i in range(len(theta)):
        theta_1 = theta.copy()
        theta_1[i] += epsilon
        theta_2 = theta.copy()
        theta_2[i] -= epsilon
        res[i] = (J(theta_1, X_b, y) - J(theta_2, X_b, y)) / (2*epsilon)  
    return res