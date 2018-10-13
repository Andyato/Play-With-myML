import numpy as np 

def accuracy_score(y_true, y_predict):
    '''准确率(acc)'''
    return np.sum(y_predict == y_true) / len(y_true)

def mean_squared_error(y_true, y_predict):
    '''均方误差(MSE)'''
    assert len(y_true) == len(y_predict), "真实值向量和预测值向量维度必须一致"
    
    return np.sum((y_predict - y_true) ** 2) / len(y_predict)

def root_mean_squared_error(y_true, y_predict):
    '''均方根误差(RMSE)'''
    return np.sqrt(mean_squared_error(y_true, y_predict))

def mean_absolute_error(y_true, y_predict):
    '''平均绝对值误差(MAE)'''
    assert len(y_true) == len(y_predict), "真实值向量和预测值向量维度必须一致"
    
    return np.sum(np.absolute(y_predict - y_true)) / len(y_predict)

def r2_score(y_true, y_predict):
    '''R平方'''
    return 1 - mean_squared_error(y_true, y_predict) / np.var(y_true)