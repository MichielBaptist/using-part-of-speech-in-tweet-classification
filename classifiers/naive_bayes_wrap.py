from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np

class naive_bayes_wrap(BaseEstimator, ClassifierMixin):  

    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self.NB = MultinomialNB(alpha, fit_prior, class_prior)

    def define_voc(self, X):        
        vectorizer = CountVectorizer()
        vectorizer.fit_transform(X)
        
        #Save the vectorizer for later use
        self.vectorizer = vectorizer
        
    def transform_text_data(self, X):
        return self.vectorizer.transform(X)
        
    def fit(self, X, y=None):
    
        #Transform 
        self.define_voc(X)
        bow_data = self.transform_text_data(X)
        
        self.NB.fit(bow_data,y)
        
        return self

    def predict(self, X, y=None):
        bow_X = self.transform_text_data(X)
        return self.NB.predict(bow_X)

    def score(self, X, y=None):
        bow_X = self.transform_text_data(X)
        return self.NB.score(bow_X, y)
    
    def get_class_distr(self, X):
        bow_X = self.transform_text_data(X)
        return self.NB.predict_proba(bow_X)
        
    def get_word_distr(self, X):
        #Very inefficient but only for trying to understand stuff        
        bow_X = self.transform_text_data(X)
        ind_to_w = self.vectorizer.get_feature_names()
        params = self.NB.feature_log_prob_
        
        line_distrs = []
        for line in bow_X:
            word_to_distr={}
            _, word_ind = line.nonzero()
            for ind in word_ind:
                word = ind_to_w[ind]
                distr= params[:,ind]
                distr= np.exp(distr)
                distr= distr/np.sum(distr)
                word_to_distr[word] = distr.tolist()
            line_distrs.append(word_to_distr)
            
        return line_distrs
        
        