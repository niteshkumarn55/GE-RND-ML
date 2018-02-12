#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 16:17:19 2018

@author: nitesh
"""

import logging
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

class CSVFilePathContants():

    #This is the Base path of all the Files
    _BASE_DOC_PATH = r"/Users/nitesh/OneDrive/Work/GE_Docs_ML/"


class CSV_NAME():

    #Fintech GE Segmented csv file
    _GE_SEGMENT_CSV = "GE_CompanyProfile.csv"

    _CLUSTER_TAG_CSV = "cluster_tag.csv"

    _CLUSTER_FILENAME_CSV = "cluster_filename.csv"

    _HC_CLUSTER_TAG_CSV = "hc_cluster_tag.csv"

    _HC_CLUSTER_FILENAME_CSV = "hc_cluster_filename.csv"

    _CLUSTER_COLUMN_CSV = "cluster_column.csv"
