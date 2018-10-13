import numpy as np 

class StandarScalar:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    def fit(self, X):
        assert X.ndim == 2, "暂时只实现二维特征空间的数据"

        self.mean_ = np.array([np.mean(X[:,i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:,i]) for i in range(X.shape[1])])

        return self
    
    def transform(self, X):
        '''将X进行均值方差归一化'''
        assert X.ndim == 2, "暂时只实现二维特征空间的数据"
        assert self.mean_ is not None and self.scale_ is not None, "must fit before tansform"
        assert X.shape[1] == len(self.scale_), "特征维度必须一致"

        resX = np.empty(shape = X.shape, dtype = float)
        for col in range(X.shape[1]):
            resX[:,col] = (resX[:,col] - self.mean_[col]) / self.scale_[col]
        return resX
