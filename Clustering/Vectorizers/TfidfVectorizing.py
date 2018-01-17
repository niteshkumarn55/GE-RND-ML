#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 16:10:19 2018

@author: nitesh
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from LoggingDetails.LogFileHandler import logs
import logging
import os
from LoggingDetails.LogPathConstant import LogFilePathContants,LogFiles
log_file = os.path.join(LogFilePathContants._BASE_LOG_FILE,
                        LogFiles._VECTORIZER_LOG_FILE)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logs().fileHandler(logging,log_file_path=log_file)
logger.addHandler(file_handler)

class VectorizingTechnique():

    def tfidf_vectorizer(self,contents):
        """

        :return:
        """
        vectorizer = TfidfVectorizer(stop_words='english',
                                     use_idf=True)
        logger.info("TF-IDF vectorizer hyperparameters {}".format(str(vectorizer)))
        X_train_vecs = vectorizer.fit_transform(contents)
        print("The TFIDF Shape : ",X_train_vecs.shape)
        print()
        # print("The vector feature names : ",vectorizer.get_feature_names())
        return X_train_vecs, vectorizer