from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import LinearSVC as SVC
from sklearn.ensemble import RandomForestClassifier as RF

import numpy as np

class average_embedding_classifier(BaseEstimator, ClassifierMixin):  

    def fit(self, X, y=None):
        print("-- Fitting the classififer!--")
        self.classifier = RF()
        self.classifier.fit(X,y)
        return self

    def predict(self, X, y=None):
        return self.classifier.predict(X)
        
    