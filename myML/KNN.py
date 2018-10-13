import numpy as np 
from math import sqrt
from collections import Counter
from .metrics import accuracy_score

class KNNclassifier:
    def __init__(self, k = 5):
        '''初始化分类器'''

        assert k >= 1, "k要不小于1"
        self.k = k
        self._X_train = None
        self._y_train = None 

    def fit(self, X_train, y_train):
        '''训练分类器'''
        assert X_train.shape[0] == y_train.shape[0], \
            "样本数与标签数要一致"
        assert self.k <= X_train.shape[0],\
            "k要小于训练样本数"

        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_test):
        '''传入测试集，返回预测值向量'''
        assert self._X_train is not None and self._y_train is not None, \
            "必须先训练模型，再进行预测"
        assert X_test.shape[1] == self._X_train.shape[1],\
            "特征空间维度不一致"
        y_predict = [self._predict(x) for x in X_test]
        return np.array(y_predict)

    def _predict(self, x):
        '''单个样本预测函数'''
        assert x.shape[0] == self._X_train.shape[1],\
            "样本空间维度不一致"
        distances = [sqrt(np.sum((x_train-x)**2)) for x_train in self._X_train]
        nearest = np.argsort(distances)
        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)
        
        return votes.most_common(1)[0][0] #返回出现最多的一个字符和频率,如[('a':5)]，加[0][0]取到该字符

    def score(self, X_test, y_test):
        '''模型评价'''
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return 'KNN(k=%d)' %self.k