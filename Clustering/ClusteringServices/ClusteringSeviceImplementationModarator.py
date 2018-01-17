#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 16:27:19 2018

@author: nitesh
"""

import logging
from LoggingDetails.LogFileHandler import logs
from LoggingDetails.LogPathConstant import LogFilePathContants,LogFiles
from CSVUtilities.CSVToDf import CsvToDataFrame
from DataMassaging.DataPreprocessingUtility.Utilities import DataUtilities, DFAnalyse
from Vectorizers.TfidfVectorizing import VectorizingTechnique as TfidfVectorizingTechnique
from Clusters.KmeansClustering.KmeansClusteringProcess import KmeansTechnique
from DataMassaging.DataPreprocessingUtility.Preprocessing.ContentProcessing import ProcessingData
import pandas as pd
import ast
import os

log_file = os.path.join(LogFilePathContants._BASE_LOG_FILE,
                        LogFiles._CLUSTERING_LOG_FILE)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logs().fileHandler(logging,log_file_path=log_file)
logger.addHandler(file_handler)

class PreClusteringModerator():

    def fetching_df(self):
        """

        :return:
        """
        CSV_Df = None
        df_independent_fields = None
        df_independent_unique_values = None

        try:
            logger.info("Getting data frame from the CSV")
            CSV_Df = CsvToDataFrame()
            try:
                if CSV_Df == None:
                    raise
                else:
                    df = CSV_Df.get_df_from_csv()
                    logger.info("CSV Loaded to the dataframe")
                    df_independent_fields = DataUtilities(df=df)

                    try:
                        if df_independent_fields == None:
                            raise
                        else:
                            df, independent_fields = df_independent_fields.show_hori_tech_from_df()
                            logger.info("removed the NaN from auto recognized independent variables")
                            logger.info("Populating the recognized independent fields {}".format(str(independent_fields
                                                                                                     )))
                            df_independent_unique_values = DFAnalyse()
                            try:
                                if df_independent_unique_values == None:
                                    raise
                                else:
                                    unique_values_dict = df_independent_unique_values.\
                                        get_unique_fields_based_on_tech_and_horizontal(df=df,
                                                                                       independent_fields=
                                                                                       independent_fields)
                                    logger.info("The unique values for the independent fields are {}"
                                                .format(str(unique_values_dict)))

                            except Exception as error:
                                logger.error("DF Analyse failed to provide the unique values {}".format(str(error)),
                                             exc_info=True)

                    except Exception as error:
                        logger.error("Data Utility filtering failed {}".format(str(error)), exc_info=True)

            except Exception as error:
                logger.error("CSV not loaded {}".format(str(error)),exc_info=True)

            return df, unique_values_dict
        except Exception as error:
            logger.error("Some Issue in fetching the data frame {}".format(str(error)),exc_info=True)

    def df_filtering(self,df=pd.DataFrame(),filter_dict=dict()):
        """

        :return:
        """
        df_filter_category = None
        try:
            df_filter_category = DFAnalyse()
            if df_filter_category == None and len(df)<=0 and len(filter_dict)<=0:
                raise
            else:
                df = df_filter_category.filter_df_by_category(df=df,dict_category_and_values=filter_dict)

            return df
        except Exception as error:
            logger.error("df filtering of category failed {}".format(str(error)),exc_info=True)

class ClusteringModerator():

    def kmeans_cluster_by_tfidf_vectorizer(self,df=pd.DataFrame()):

        # Tfidf vectorization process to get the vectors for bow for all documents with in the doctype
        try:
            vector_bow = TfidfVectorizingTechnique()
            """kmeans_Clustering is performed without tfidf pipelined in the count vectorizing matrix"""
            contents = df['processed_text'].tolist()
            company_id = df['ID'].tolist()
            X_train_vecs, vectorizer = vector_bow.tfidf_vectorizer(contents)
            logger.info("Preprocessed data got converted into features -> vectors")
            kmeans = KmeansTechnique()
            # Kmeans without the LSA dimensionality reduction
            cluster_and_terms_map, cluster_and_count_of_docs = kmeans.kmeans_cluster_technique(number_cluster=8,
            X_train_vecs=X_train_vecs, vectorizer=vectorizer, filenames=company_id, contents=contents,
            is_dimension_reduced=False)

            #TODO: See how to enble the Vectorizer.get_feature_names() for the pipelined tfidf vectors.
            #TODO: Follow the commented steps in the mini_kmeans_cluster_by_count_vectorizer() above, after figuring out how to get the get_feature_names()
            return cluster_and_terms_map, cluster_and_count_of_docs
        except Exception as error:
            logger.error("Kmeans clustering failed {}".format(str(error)))

class ExtraTools():
    def filter_input(self):
        filter_dict = dict()
        while True:
            print("filtered dict... press 1 to continue 2 to break")
            temp = input()
            if temp=="1":

                print("Enter the field name")
                key = str(input())
                print(key)
                print("Enter the value")
                value = str(input())
                print(value)
                val = list()
                if key in filter_dict:
                    val = filter_dict.get(key)
                    val.append(value)
                    filter_dict[key] = val
                else:
                    val.append(value)
                    filter_dict[key] = val

            else:
                break;
        return filter_dict

if __name__ == '__main__':
    pcm = PreClusteringModerator()
    df, unique_values_dict=pcm.fetching_df()

    d = {'horizontal': ['Fintech'], 'technology_segment_1': ['Digital Banking']}
    df = pcm.df_filtering(df=df,filter_dict=d)
    df = ProcessingData().get_text_without_stem(df=df)

    cm = ClusteringModerator()
    cm.kmeans_cluster_by_tfidf_vectorizer(df=df)



