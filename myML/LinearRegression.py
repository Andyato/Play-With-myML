import numpy as np 
from .metrics import r2_score

class LinearRegression:
    def __init__(self):
        self.coef_ = None             #特征前系数向量
        self.interception_ = None     #截距
        self._theta = None

    def fit_normal(self, X_train, y_train):
        '''训练模型'''
        assert X_train.shape[0] == y_train.shape[0], "特征和结果样本数要一致"

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.coef_ = self._theta[1:]
        self.interception_ = self._theta[0]

        return self

    def fit_gd(self, X_train, y_train, eta = 0.01, n_iters = 1e4, epsilon=1e-8):
        '''使用梯度下降法训练模型'''
        assert X_train.shape[0] == y_train.shape[0], "特征和结果样本数要一致"

        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta) ** 2)) / len(y)
            except:
                return float('inf')
        
        def dJ(theta, X_b, y):
            # res = np.empty(len(theta))
            # res[0] = np.sum(X_b.dot(theta) - y)
            # for i in range(1, len(theta)):
            #     res[i] = np.sum((X_b.dot(theta) - y).dot(X_b[:,i]))
            # return res * 2 / len(y)
            return X_b.T.dot(X_b.dot(theta) - y) * 2. / len(y)   #向量化实现

        def gradient_descent(X_b, y, initial_theta, eta, n_iters, epsilon):
            theta = initial_theta
            cur_iters = 0

            while cur_iters < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = last_theta - eta * gradient
                if abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon:
                    break
                cur_iters += 1

            return theta

        X_b = np.hstack([np.ones((len(X_train),1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters, epsilon)
        self.coef_ = self._theta[1:]
        self.interception_ = self._theta[:1]

        return self

    def fit_sgd(self, X_train, y_train, epoch=5, t0=5, t1=50):
        '''使用随机梯度下降法训练模型'''
        assert X_train.shape[0] == y_train.shape[0], "特征和结果样本数要一致"
        assert epoch >= 1, "至少把所由样本看一遍"

        def dJ_sgd(theta, X_b_i, y_i):
            return X_b_i.T.dot(X_b_i.dot(theta) - y_i) * 2

        def sgd(X_b, y, initial_theta, epoch, t0, t1):
            
            def learning_rate(t):
                return t0 / (t + t1)

            theta = initial_theta
            m = len(X_b)

            for i in range(epoch):
                index_shuffled = np.random.permutation(m)
                X_b_new = X_b[index_shuffled]
                y_new = y[index_shuffled]
                for j in range(m):
                    gradient = dJ_sgd(theta, X_b_new[j], y_new[j])
                    theta -= learning_rate(i*m + j) * gradient

            return theta

        X_b = np.hstack([np.ones((len(y_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = sgd(X_b, y_train, initial_theta, epoch, t0, t1)
        self.coef_ = self._theta[1:]
        self.interception_ = self._theta[:1]

        return self


    def predict(self, X_predict):
        '''预测'''
        assert self.coef_ is not None and self.interception_ is not None, \
            "must fit before predict"
        assert X_predict.shape[1] == self.coef_.shape[0], "特征数和特征系数数量要一致"
        
        X_b = np.hstack([np.ones((len(X_predict),1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        '''模型评价'''
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"