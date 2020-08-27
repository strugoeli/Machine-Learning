import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from numpy.linalg import multi_dot
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

"""
   A Model class - a general class for classifier models for this exercise.

   """


class Model:
    """
    Type of scores
    """
    score_keys = {'num_samples', 'accuracy', 'error', 'FPR', 'TPR', 'precision', 'recall'}

    def __init__(self, pos=1, neg=-1):
        """

        :param pos:
        :param neg:
        """
        self.w = 0
        self.to_add_ones = True
        self.pos_val = pos
        self.neg_val = neg

    def fit(self, X, y):
        """
        Given a training set a X dxm and y in {pos_val,neg_val} this methods learns the parameters of
        the model and stores the trained model
        :param X: numpy array training set a X dxm
        :param y: numpy array y
        """
        return

    def predict(self, X):
        """
        Given an unlabeled test set X dxm', predicts the label of each sample.
        :param X:numpy array dxm'
        :return:a vector of predicted labels y of size m'
        """
        X = self.add_ones(X) if self.to_add_ones else X
        return np.sign(np.dot(self.w, X))

    def score(self, X, y):
        """
        Given an unlabeled test set X, and the true labels y of the test set and returns a dictionary with fields:
        {'num_samples', 'accuracy', 'error', 'FPR', 'TPR', 'precision', 'recall'}
        :param X: numpy array dxm'
        :param y: numpy array of size m'
        :return: a dictionary
        """
        scores = dict()
        prediction = self.predict(X)

        # Calculating relevant parameters for the scores
        P = np.sum(y == self.pos_val)
        N = np.sum(y == self.neg_val)

        TP = np.sum((prediction == y) & (y == self.pos_val))
        TN = np.sum((prediction == y) & (y == self.neg_val))

        FP = np.sum((prediction != y) & (prediction == self.pos_val))
        FN = np.sum((prediction != y) & (prediction == self.neg_val))

        scores['num_samples'] = y.size
        scores['accuracy'] = (TN + TP) / y.size
        scores['error'] = (FP + FN) / y.size
        scores['FPR'] = FP / N
        scores['TPR'] = TP / P
        scores['precision'] = TP / (TP + FP)
        scores['recall'] = TP / P
        return scores

    def get_from_score(self, X, y, key):
        """
        Get a single score by the given key
        """
        if key in self.score_keys:
            return self.score(X, y)[key]
        print("Wrong score key")

    def get_type(self):
        """
        :return:The type of the model
        """
        return ''

    @staticmethod
    def add_ones(X):
        """
        Add row of ones to the given set
        :param X: numpy array
        :return:  given X with extra vector of ones
        """
        return np.append(X, np.ones([1, X.shape[1]]), axis=0)


class Perceptron(Model):
    """
    Implementation of Perceptron classifier
    """

    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        X = self.add_ones(X)
        self.w = np.zeros([X.shape[0]])
        A = np.multiply(X.T, y[:, np.newaxis])

        while True:
            val = np.dot(A, self.w)
            i = np.argmin(val)
            if val[i] <= 0:
                self.w += X[:, i] * y[i]
            else:
                return

    def get_type(self):
        return 'Perceptron'


def delta_function(X, sigma_inv, mu, p):
    return multi_dot([X.T, sigma_inv, mu]) - (.5 * multi_dot([mu.T, sigma_inv, mu])) + np.log(p)


class LDA(Model):
    """
    Implementation of LDA classifier for multivariate normal distribution of dim 2
    """

    def __init__(self):
        super().__init__()
        self.prior = 0
        self.mu = 0
        self.sigma = 0
        self.to_add_ones = False

    def fit(self, X, y):
        n = y.size

        # MLE estimator for the means
        mean_pos = np.mean(X[:, y == self.pos_val], axis=1)
        mean_neg = np.mean(X[:, y == self.neg_val], axis=1)
        self.mu = mean_pos, mean_neg

        # MLE estimator for the prior
        p_pos = np.sum(y == self.pos_val) / n
        p_neg = np.sum(y == self.neg_val) / n
        self.prior = p_pos, p_neg

        N = np.sum(y == self.neg_val)
        P = np.sum(y == self.pos_val)

        # Using "pooled covariance" to estimate sigma
        if N - 1 and P - 1:
            cov_pos = np.float(N - 1) * np.cov(X[:, y == self.pos_val])
            cov_neg = np.float(P - 1) * np.cov(X[:, y == self.neg_val])
            self.sigma = (cov_neg + cov_pos) / (n - 2)
        else:
            self.sigma = np.cov(X, bias=True)

    def predict(self, X):

        p_pos, p_neg = self.prior
        mean_pos, mean_neg = self.mu
        sigma_inv = np.linalg.inv(self.sigma)

        return np.sign(delta_function(X, sigma_inv, mean_pos, p_pos) - delta_function(X, sigma_inv, mean_neg, p_neg))

    def get_type(self):
        return 'LDA'


class SVM(Model):
    """
    Implementation of Hard SVM classifier using sklearn
    """

    def __init__(self, pos=1, neg=-1, reg=1e10):
        super().__init__(pos, neg)
        self.svm = svm.SVC(C=reg, kernel="linear")

    def fit(self, X, y):
        X = self.add_ones(X)

        self.svm.fit(X.T, y)
        self.w = np.squeeze(self.svm.coef_)

    def predict(self, X):
        X = self.add_ones(X)

        return self.svm.predict(X.T)

    def get_type(self):
        return 'SVM'


class SoftSVM(Model):
    """
    Implementation of Hard SVM classifier using sklearn

    """

    def __init__(self, pos=1, neg=0, reg=1e10):
        super().__init__(pos, neg)
        self.s_svm = svm.LinearSVC(C=reg, loss='hinge')

    def fit(self, X, y):
        X = self.add_ones(X)

        self.s_svm.fit(X.T, y)
        self.w = np.squeeze(self.s_svm.coef_)

    def predict(self, X):
        X = self.add_ones(X)

        return self.s_svm.predict(X.T)

    def get_type(self):
        return 'SoftSVM'


class Logistic(Model):
    """
    Implementation of Logistic classifier using sklearn

    """

    def __init__(self, pos=1, neg=0):
        super().__init__(pos, neg)
        self.logistic = LogisticRegression(solver='liblinear')

    def fit(self, X, y):
        X = self.add_ones(X)
        self.logistic.fit(X.T, y)
        self.w = np.squeeze(self.logistic.coef_)

    def predict(self, X):
        X = self.add_ones(X)
        return self.logistic.predict(X.T)

    def get_type(self):
        return 'Logistic'


class DecisionTree(Model):
    """
    Implementation of Logistic classifier using sklearn used for Mnist data

    """

    def __init__(self, pos=1, neg=0):
        super().__init__(pos, neg)
        self.tree = DecisionTreeClassifier(splitter="random", max_depth=3, min_samples_split=2)

    def fit(self, X, y):
        self.tree.fit(X.T, y)

    def predict(self, X):
        return self.tree.predict(X.T)

    def get_type(self):
        return 'DecisionTree'


class KNeighbors(Model):
    """
      Implementation of KNeighbors classifier using sklearn

    """

    def __init__(self, num_neighbors, pos=1, neg=0):
        super().__init__(pos, neg)
        self.knn = KNeighborsClassifier(n_neighbors=num_neighbors)

    def fit(self, X, y):
        X = self.add_ones(X)
        self.knn.fit(X.T, y)

    def predict(self, X):
        X = self.add_ones(X)
        return self.knn.predict(X.T)

    def get_type(self):
        return 'KNeighbors'
