#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 16:08:09 2018

@author: nitesh
"""

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
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

    def count_vectorizer(self,contents):
        """
        bags of words representation using the CountVectorizer.
        CountVectorizer - Convert a collection of text documents to a matrix of token counts.

        The bags of words representation implies that n_features is the number of distinct words in the corpus:
        this number is typically larger than 100,000.

        If n_samples == 10000, storing X as a numpy array of type float32 would require
        10000 x 100000 x 4 bytes = 4GB in RAM which is barely manageable on today’s computers.

        Fortunately, most values in X will be zeros since for a given document less than a
        couple thousands of distinct words will be used. For this reason we say that bags of words
        are typically high-dimensional sparse datasets. We can save a lot of memory by only storing the
        non-zero parts of the feature vectors in memory.

        :return: Vector shape - for each document #i, count the number of occurrences of each word w and store it in X[i, j]
        as the value of feature #j where j is the index of word w in the dictionary
        """
        vectorizer = CountVectorizer()
        X_train_vecs = vectorizer.fit_transform(contents)
        print("The count of bow : ", X_train_vecs.shape)
        return X_train_vecs, vectorizer

    def count_tfidf_vectorizer(self,contents):
        """
        Occurrence count is a good start but there is an issue:
        longer documents will have higher average count values than shorter documents,
        even though they might talk about the same topics.

        To avoid these potential discrepancies it suffices to divide the number of
        occurrences of each word in a document by the total number of words in the document:
        these new features are called tf for Term Frequencies.

        Another refinement on top of tf is to downscale weights for words that occur in many documents
        in the corpus and are therefore less informative than those that occur only in a smaller portion of the corpus.

        This downscaling is called tf–idf for “Term Frequency times Inverse Document Frequency”.

        The bags of words representation implies that n_features is the number of distinct words in the corpus:
        this number is typically larger than 100,000.

        If n_samples == 10000, storing X as a numpy array of type float32 would require
        10000 x 100000 x 4 bytes = 4GB in RAM which is barely manageable on today’s computers.

        Fortunately, most values in X will be zeros since for a given document less than a
        couple thousands of distinct words will be used. For this reason we say that bags of words
        are typically high-dimensional sparse datasets. We can save a lot of memory by only storing the
        non-zero parts of the feature vectors in memory.

        CountVectorizer - Convert a collection of text documents to a matrix of token counts
        TfidfVectorizer - Convert a collection of raw documents to a matrix of TF-IDF features.

        :return: Vector shape - for each document #i, count the number of occurrences of each word w and store it in X[i, j]
        as the value of feature #j where j is the index of word w in the dictionary
        """
        count_vect = CountVectorizer()
        vectorizer = make_pipeline(count_vect,TfidfTransformer())
        X_train_vecs = vectorizer.fit_transform(contents)
        print("The count of bow : ", X_train_vecs.shape)
        return X_train_vecs, vectorizer

