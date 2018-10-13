import numpy as np 
from .metrics import r2_score

class SimpleLinearRegression1:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert x_train.ndim == 1, "一元线性回归"
        assert len(x_train) == len(y_train), "the size of x_train must equal to the size of y_train"
    
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        numerator = 0.0
        denominator = 0.0
        for x_i, y_i in zip(x_train, y_train):
            numerator += (x_i - x_mean) * (y_i - y_mean)
            denominator += (x_i - x_mean) ** 2
        self.a_ = numerator / denominator
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        assert x_predict.ndim == 1, "一元线性回归"
        assert self.a_ is not None and self.b_ is not None, "must fit before predict"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression1()"

class SimpleLinearRegression:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert x_train.ndim == 1, "一元线性回归"
        assert len(x_train) == len(y_train), "the size of x_train must equal to the size of y_train"
    
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        numerator = (x_train - x_mean).dot(y_train - y_mean)
        denominator = (x_train - x_mean).dot(x_train - x_mean)
        
        self.a_ = numerator / denominator
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        assert x_predict.ndim == 1, "一元线性回归"
        assert self.a_ is not None and self.b_ is not None, "must fit before predict"

        return np.array([self._predict(x) for x in x_predict])

    def score(self, x_test, y_test):
        '''模型评价'''
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)

    def _predict(self, x_single):
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression()(using np.dot())"