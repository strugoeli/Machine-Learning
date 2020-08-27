"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Gad Zalcberg
Date: February, 2019

"""
import numpy as np


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None] * T  # list of base learners
        self.w = np.zeros(T)  # weights
        self.D = None

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """
        m = len(y)
        D = np.ones(m) / m
        for t in range(self.T):
            self.h[t] = self.WL(D, X, y)
            predictions = self.h[t].predict(X)
            epsilon_t = np.sum(D[y != self.h[t].predict(X)])
            self.w[t] = 0.5 * np.log((1 / epsilon_t) - 1)
            curr_D = D * np.exp(-self.w[t] * y * predictions)
            D = np.divide(curr_D, np.sum(curr_D))
        self.D = D

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        gen = (np.dot(self.w[t], self.h[t].predict(X)) for t in range(max_t))
        y_hat = np.sum(gen, axis=0)
        return np.sign(y_hat).astype(int)

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the correct predictions when predict only with max_t weak learners (float)
        """
        return np.count_nonzero(self.predict(X, max_t) != y) / len(y)
