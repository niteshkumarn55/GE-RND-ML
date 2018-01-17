#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 16:09:25 2018

@author: nitesh
"""

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from eli5.sklearn import InvertableHashingVectorizer
from sklearn.pipeline import make_pipeline
from LoggingDetails.LogFileHandler import logs
import logging
import os
from DataMassaging.DataPreprocessingUtility.PathConstant import DescribeFilePathContants
log_file = os.path.join(DescribeFilePathContants._BASE_LOG_FILE,
                                            DescribeFilePathContants._DATA_MASSAGE_LOG_FILE)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logs().fileHandler(logging,log_file_path=log_file)
logger.addHandler(file_handler)

class VectorizingTechnique():

    def hashing_vectorizer(self,contents):
        """
        Convert a collection of text documents to a matrix of token occurrences.

        It turns a collection of text documents into a scipy.sparse matrix holding token occurrence counts (or binary occurrence information),
        possibly normalized as token frequencies if norm=’l1’ or projected on the euclidean unit sphere if norm=’l2’.

        HashingVectorizer hashes word occurrences to a fixed dimensional space, possibly with collisions.
        The word count vectors are then normalized to each have l2-norm equal to one (projected to the euclidean unit-ball)
        which seems to be important for k-means to work in high dimensional space.

        :param contents:
        :return:
        """
        vectorizer = HashingVectorizer(n_features=10000,
                                       stop_words='english',
                                       non_negative=False, norm='l2',
                                       binary=False)
        ivec = InvertableHashingVectorizer(vectorizer)
        ivec.fit(contents)
        X_train_vecs = vectorizer.fit_transform(contents)
        print("The Hashing Vectorizer shape is : ",X_train_vecs.shape)
        return X_train_vecs, ivec

    def hash_tfidf_vectorizer(self,contents):
        """
        HashingVectorizer does not provide IDF weighting as this is a stateless model (the fit method does nothing).
        When IDF weighting is needed it can be added by pipelining its output to a TfidfTransformer instance.
        :param contents:
        :return:
        """
        hasher = HashingVectorizer(stop_words='english', non_negative=True,
                                   norm=None, binary=False)
        vectorizer = make_pipeline(hasher, TfidfTransformer())
        X_train_vecs = vectorizer.fit_transform(contents)
        print("The hash vector and tfidf vector shape : ",X_train_vecs.shape)
        return X_train_vecs, vectorizer