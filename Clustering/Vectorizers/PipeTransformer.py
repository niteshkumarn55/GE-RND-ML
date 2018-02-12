#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 16:10:19 2018

@author: nitesh
"""
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer,CountVectorizer,TfidfTransformer
from sklearn.pipeline import Pipeline

class PipeVecToTransformer():
    """

    """
    def pipe_vec_to_tfidf_transformer(self,vectorizer=None, classifier=None, classifier_name=None, X_train=None,y_train=None):
        """

        :param vectorizer: Takes the vectorized of the text
        :param classifier: get the object of the classifier you are using ex: classifier=dt [where dt = DecisionTree()]
        :param classifier_name: name of the classifier
        :param X_train: training data
        :param y_train:
        :return:
        """
        #X_train = vectorizer.fit_transform(train_data['processed_text'].tolist())
        #X_test = vectorizer.transform(test_data['processed_text'].tolist())

        vec_clf = Pipeline([('vec', vectorizer),('tfidf', TfidfTransformer()), (classifier_name, classifier)])
        vec_clf = vec_clf.fit(X_train,y_train)

        return vec_clf