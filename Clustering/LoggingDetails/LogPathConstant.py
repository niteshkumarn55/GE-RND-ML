#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 12:12:19 2018

@author: nitesh
"""

import logging
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

class LogFilePathContants():

    _BASE_LOG_FILE = r"/Users/nitesh/OneDrive/Work/GE_Python_Workspace/Clustering logs/"

class LogFiles():

    _DATA_MASSAGE_LOG_FILE = r"data_massage_log.log"

    _CLUSTERING_LOG_FILE = r"classifier_log.log"

    _CSV_LOG_FILE = r"csv_log.log"

    _VECTORIZER_LOG_FILE = r"vectorizer_log.log"


