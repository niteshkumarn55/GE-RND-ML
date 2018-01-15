#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 13:04:36 2017

@author: nitesh
"""


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

class VectorizersTemplates():
    """

    """
    def tfidf_vectorizer_templates(self):
        """

        :return:
        """
        vectorizer_tuple = tuple()
        vectorizer_name = 'tfidf_vec'
        doc_vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                         stop_words='english')
        vectorizer_tuple = (vectorizer_name, doc_vectorizer)
        return doc_vectorizer, vectorizer_tuple

    def count_vectorizer_template(self):
        """

        :return:
        """
        vectorizer_tuple = tuple()
        vectorizer_name = 'count_vec'
        doc_vectorizer = CountVectorizer()
        vectorizer_tuple = (vectorizer_name,doc_vectorizer)
        return doc_vectorizer, vectorizer_tuple

    def dict_vectorizer_template(self):
        """

        :return:
        """
        vectorizer_tuple = tuple()
        vectorizer_name = 'dict_vec'
        doc_vectorizer = DictVectorizer()
        vectorizer_tuple = (vectorizer_name, doc_vectorizer)
        return doc_vectorizer, vectorizer_tuple

    def Hash_vectorizer_template(self):
        """

        :return:
        """
        vectorizer_tuple = tuple()
        vectorizer_name = 'hash_vec'
        doc_vectorizer = HashingVectorizer()
        vectorizer_tuple = (vectorizer_name, doc_vectorizer)
        return doc_vectorizer, vectorizer_tuple

class TransformersTemplates():
    """

    """

    def tfidf_trasformer_template(self):
        """

        :return:
        """
        transformer_tuple = tuple()
        transformer_name = 'tfidf'
        doc_transformer = TfidfTransformer()
        transformer_tuple = (transformer_name,doc_transformer)
        return doc_transformer, transformer_tuple

class ClassfierVectorizer():
    """

    """
    def get_classified_vectorizer(self,vectorizer_tuple=None, transformer_tuple = None, classifier_tuple=None, X_train=None,y_train=None):
        """
        get the vectorized classifier model for the classifier and the vectorizer that you provide
        :param vectorizer_tuple: tuple of name and the vectorizer by which it needs to me vectorized for the x and y trains
        :param transformer_tuple: tuple of name and transfomer by which it needs to be transformed
        :param classifier_tuple: tuple of name of the classifier and the classifier for which the model should be built
        :param X_train: X training dataset
        :param y_train: y (labels or categories) for the training the X for the model
        :return:
        """

        #X_train = vectorizer.fit_transform(train_data['processed_text'].tolist())
        #X_test = vectorizer.transform(test_data['processed_text'].tolist())

        if transformer_tuple == None:
            vec_clf = Pipeline([vectorizer_tuple, classifier_tuple])
        else:
            vec_clf = Pipeline([vectorizer_tuple, transformer_tuple, classifier_tuple])
        vec_clf = vec_clf.fit(X_train,y_train)

        return vec_clf