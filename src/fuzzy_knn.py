from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.metrics import accuracy_score

class FuzzyKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def predict(self, X):
        y_pred = []
        for x in X:
            distances = np.linalg.norm(self.X - x, axis=1)
            idx = np.argsort(distances)[:self.k]
            votes = self.y[idx]
            y_pred.append(max(set(votes), key=list(votes).count))
        return y_pred

    def score(self, X, y):
        preds = self.predict(X)
        return accuracy_score(y, preds)
