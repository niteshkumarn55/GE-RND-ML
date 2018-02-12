#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 16:27:19 2018

@author: nitesh
"""
import os
import sys
# print(os.path.abspath(os.path.join(os.getcwd(),os.pardir)))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),os.pardir)))

_cluster_number = None
_cluster_stem = None
_percentage_of_cluster_bestfit = None



import logging
from ClusteringServices.ServiceUtilities import ExtraTools
from LoggingDetails.LogFileHandler import logs
from LoggingDetails.LogPathConstant import LogFilePathContants,LogFiles
from CSVUtilities.CSVToDf import CsvToDataFrame
from CSVUtilities.CSVAndDFToDict import CSVToDictionaryMapping
from CSVUtilities.DFToCSV import DataframeToCSV
from DataMassaging.DataPreprocessingUtility.Utilities import DataUtilities, DFAnalyse
from Vectorizers.TfidfVectorizing import VectorizingTechnique as TfidfVectorizingTechnique
from Clusters.KmeansClustering.KmeansClusteringProcess import KmeansTechnique
from DataMassaging.DataPreprocessingUtility.Preprocessing.ContentProcessing import ProcessingData
from Clusters.BestFitCluster.BestFitNumberOfCluster import BestFitClusterTechnique,RuleBasedBestFitClusters
from Clusters.HierarchicalClustering.HierarchicalClusteringProcess import HierarchyTechnique
import pandas as pd
import os


log_file = os.path.join(LogFilePathContants._BASE_LOG_FILE,
                        LogFiles._CLUSTERING_LOG_FILE)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logs().fileHandler(logging,log_file_path=log_file)
logger.addHandler(file_handler)

try:
    if len(sys.argv) == 2:
        _cluster_number = sys.argv[1]

        if(_cluster_number!='None'):
            _cluster_number = int(_cluster_number)

        print('the cluster number is: ',_cluster_number,' and the type is ',type(_cluster_number))
    elif len(sys.argv) == 3:
        _cluster_number = int(sys.argv[1])
        _cluster_stem = str(sys.argv[2])

        if (_cluster_number != 'None'):
            _cluster_number = int(_cluster_number)

        print('The cluster number is: ',_cluster_number,' and the type is ',type(_cluster_number))
        print('the cluster stem or not:',_cluster_stem, ' and the type is ', type(_cluster_stem))
    elif len(sys.argv) == 4:
        _cluster_number = sys.argv[1]
        _cluster_stem = str(sys.argv[2])
        _percentage_of_cluster_bestfit = sys.argv[3]

        if (_cluster_number != 'None'):
            _cluster_number = int(_cluster_number)

        if(_percentage_of_cluster_bestfit!='None'):
            _percentage_of_cluster_bestfit = int(_percentage_of_cluster_bestfit)

        print('The cluster number is: ', _cluster_number, ' and the type is ',type(_cluster_number))
        print('the cluster stem or not:', _cluster_stem,' and the type is ',type(_cluster_stem))
        print('the percentage is: ',_percentage_of_cluster_bestfit,
              ' and the type is ',type(_percentage_of_cluster_bestfit))
except Exception as error:
    logger.error("Some issue taking the argument from the terminal {}".format(str(error)))


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
                    raise Exception
                else:
                    df = CSV_Df.get_df_from_csv()
                    logger.info("CSV Loaded to the dataframe")
                    df_independent_fields = DataUtilities(df=df)

                    try:
                        if df_independent_fields == None:
                            raise Exception
                        else:
                            df, independent_fields = df_independent_fields.show_hori_tech_from_df()
                            logger.info("removed the NaN from auto recognized independent variables")
                            logger.info("Populating the recognized independent fields {}".format(str(independent_fields
                                                                                                     )))
                            df_independent_unique_values = DFAnalyse()
                            try:
                                if df_independent_unique_values == None:
                                    raise Exception
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
            This function take the dataframe and the filter dict and filter the df with respect to the filter_dict
        :param df: df is from the csv which is in _GE_SEGMENT_CSV = "GE_CompanyProfile.csv" from the csv constant path.
        :param filter_dict: this is taken from the _CLUSTER_COLUMN_CSV = "cluster_column.csv" if you provide filter as
        "all": "full" the no filter is done takes all the fields for the classification. if you specify a value it will be
        filtered for the same
        :return: This will return the final filtered df
        """
        df_filter_category = None
        try:
            df_filter_category = DFAnalyse()
            if df_filter_category == None and len(df)<=0 and len(filter_dict)<=0:
                raise Exception
            else:
                df = df_filter_category.filter_df_by_category(df=df,dict_category_and_values=filter_dict)

            return df
        except Exception as error:
            logger.error("df filtering of category failed {}".format(str(error)),exc_info=True)

class ClusteringModerator():

    def kmeans_cluster_by_tfidf_vectorizer(self,df=pd.DataFrame(),cluster_number=1):
        """

        :param df:
        :return:
        """

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
            cluster_and_terms_map, cluster_and_count_of_docs,dict_of_cluster_and_tags,dict_of_cluster_and_filename = \
            kmeans.kmeans_cluster_technique(number_cluster=cluster_number,
            X_train_vecs=X_train_vecs, vectorizer=vectorizer, filenames=company_id, contents=contents,
            is_dimension_reduced=False)

            #TODO: See how to enble the Vectorizer.get_feature_names() for the pipelined tfidf vectors.
            #TODO: Follow the commented steps in the mini_kmeans_cluster_by_count_vectorizer() above, after figuring out how to get the get_feature_names()
            print("dict_of_cluster_and_tags:", dict_of_cluster_and_tags)
            print("dict_of_cluster_and_filename:",dict_of_cluster_and_filename)
            return dict_of_cluster_and_tags, dict_of_cluster_and_filename
        except Exception as error:
            logger.error("Kmeans clustering failed {}".format(str(error)))

    def hierachical_dendogram_bestfit(self,df=pd.DataFrame()):
        """

        :return:
        """
        # Tfidf vectorization process to get the vectors for bow for all documents with in the doctype
        try:
            vector_bow = TfidfVectorizingTechnique()
            """kmeans_Clustering is performed without tfidf pipelined in the count vectorizing matrix"""
            contents = df['processed_text'].tolist()
            company_id = df['ID'].tolist()
            X_train_vecs, vectorizer = vector_bow.tfidf_vectorizer(contents)
            logger.info("Preprocessed data got converted into features for dendogram-> vectors")
            dendrogram = HierarchyTechnique()
            dendrogram.hierarchy_dendrogram(X_train_vecs=X_train_vecs,filenames=company_id)
            logger.info("Hierarchy dendrogram is created")
        except Exception as error:
            logger.error("Dendrogram failed {}".format(str(error)))

    def hierarchical_cluster_by_tfidf_vectorizer(self,df=pd.DataFrame(),cluster_number=1):
        """

        :param df:
        :return:
        """
        # Tfidf vectorization process to get the vectors for bow for all documents with in the doctype
        try:
            vector_bow = TfidfVectorizingTechnique()
            """Hierarchy_Clustering is performed without tfidf pipelined in the count vectorizing matrix"""
            contents = df['processed_text'].tolist()
            company_id = df['ID'].tolist()
            X_train_vecs, vectorizer = vector_bow.tfidf_vectorizer(contents)
            logger.info("Preprocessed data got converted into features -> vectors")
            hierarchy = HierarchyTechnique()
            cluster_and_terms_map, cluster_and_count_of_docs, dict_of_cluster_and_tags, dict_of_cluster_and_filename = \
                hierarchy.hierarchy_cluster_technique(number_cluster=cluster_number,
                                                X_train_vecs=X_train_vecs, vectorizer=vectorizer, filenames=company_id,
                                                contents=contents,
                                                is_dimension_reduced=False)

            print("dict_of_cluster_and_tags:", dict_of_cluster_and_tags)
            print("dict_of_cluster_and_filename:", dict_of_cluster_and_filename)
            return dict_of_cluster_and_tags, dict_of_cluster_and_filename
        except Exception as error:
            logger.error("Hierarchical clustering failed {}".format(str(error)))

    def bestfit_kmeans(self, df=pd.DataFrame()):
        """
        :param df:
        :return:
        """
        # Tfidf vectorization process to get the vectors for bow for all documents with in the doctype
        try:
            vector_bow = TfidfVectorizingTechnique()
            """kmeans_Clustering is performed without tfidf pipelined in the count vectorizing matrix"""
            contents = df['processed_text'].tolist()
            X_train_vecs, vectorizer = vector_bow.tfidf_vectorizer(contents)
            logger.info("Preprocessed data got converted into features for bestfit-> vectors")
            bestfit_kmeans = BestFitClusterTechnique()
            bestfit_kmeans.cluster_by_elbow_method_for_kmeans(max_cluster_iter=300,X_train_matrix=X_train_vecs)
            logger.info("Elbow method is complete and the graph is populated")
        except Exception as error:
            logger.error("Best fit of kmeans failed {}".format(str(error)))


def initiate_kmeans(k_number=None):
    pcm = PreClusteringModerator()
    bfit = RuleBasedBestFitClusters()
    util = ExtraTools()
    df_to_csv = DataframeToCSV()
    csv_and_dict = CSVToDictionaryMapping()
    df, unique_values_dict = pcm.fetching_df()
    filtered_tech_dict = util.get_filtered_dict()

    df = pcm.df_filtering(df=df, filter_dict=filtered_tech_dict)

    count_of_categories = util.get_number_of_categories(unique_values_dict_categories=unique_values_dict)

    # Dataframe after the preprocessing --------------------
    if(_cluster_stem=='stem'): #Steming happens only when you ask it from cmd prompt
        logger.info('The process is stem from the cmd')
        df = ProcessingData().get_complete_text(df=df) #stemming the text

    else:
        logger.info('The process is not stemed')
        df = ProcessingData().get_text_without_stem(df=df)  # No stemming is done for the text

    if(k_number==None):
        if(_percentage_of_cluster_bestfit=='None' or _percentage_of_cluster_bestfit==None): #If the cluster number is None then it takes a default percentage which is 60%
            logger.info('The lenght if the df is {}'.format(str(len(df))))
            logger.info('Default percentage is consider which is 60%')
            k_number=bfit.best_fit_clusters_based_df_count(count_of_categories=count_of_categories,df_count=len(df))


        else:#If Percentage is given in the cmd then percentage is override from the default percentage which is 60%
            logger.info('The lenght if the df is {}'.format(str(len(df))))
            logger.info('Default percentage is overridden for the percentage given {}'
                        .format(str(_percentage_of_cluster_bestfit)))
            k_number=bfit.best_fit_clusters_based_df_count(count_of_categories=count_of_categories,df_count=len(df),
                                                           percentage_for_cluster_number=_percentage_of_cluster_bestfit)


    logger.info("The total number of cluster is %s for the records %s",str(k_number),str(len(df)))
    cm = ClusteringModerator()
    dict_of_cluster_and_tags, dict_of_cluster_and_filename = cm.kmeans_cluster_by_tfidf_vectorizer(df=df,
                                                                                            cluster_number=k_number)

    # cm.bestfit_kmeans(df=df)

    dict_tag_df = csv_and_dict.dict_to_df(dict_of_cluster_and_tags)
    df_to_csv.load_df_to_csv(df=dict_tag_df, cluster_csv="cluster_tag",algo="kmeans")

    dict_filename_df = csv_and_dict.dict_to_df(dict_of_cluster_and_filename)
    df_to_csv.load_df_to_csv(df=dict_filename_df, cluster_csv="cluster_filename",algo="kmeans")

def initiate_hierarchical_dendrogram():
    pcm = PreClusteringModerator()
    util = ExtraTools()
    df, unique_values_dict = pcm.fetching_df()
    filtered_tech_dict = util.get_filtered_dict()

    df = pcm.df_filtering(df=df, filter_dict=filtered_tech_dict)
    df = ProcessingData().get_text_without_stem(df=df)

    cm = ClusteringModerator()
    cm.hierachical_dendogram_bestfit(df=df)

def initiate_hierarchy(hc_number=1):
    pcm = PreClusteringModerator()
    util = ExtraTools()
    df_to_csv = DataframeToCSV()
    csv_and_dict = CSVToDictionaryMapping()
    df, unique_values_dict = pcm.fetching_df()
    filtered_tech_dict = util.get_filtered_dict()

    df = pcm.df_filtering(df=df, filter_dict=filtered_tech_dict)
    df = ProcessingData().get_text_without_stem(df=df)

    cm = ClusteringModerator()
    dict_of_cluster_and_tags, dict_of_cluster_and_filename = cm.hierarchical_cluster_by_tfidf_vectorizer(df=df,
                                                                                            cluster_number=hc_number)

    dict_tag_df = csv_and_dict.dict_to_df(dict_of_cluster_and_tags)
    df_to_csv.load_df_to_csv(df=dict_tag_df, cluster_csv="cluster_tag",algo="hierarchy")

    dict_filename_df = csv_and_dict.dict_to_df(dict_of_cluster_and_filename)
    df_to_csv.load_df_to_csv(df=dict_filename_df, cluster_csv="cluster_filename",algo="hierarchy")


def initiate_bestfit():
    pcm = PreClusteringModerator()
    util = ExtraTools()
    df, unique_values_dict = pcm.fetching_df()
    filtered_tech_dict = util.get_filtered_dict()

    df = pcm.df_filtering(df=df, filter_dict=filtered_tech_dict)
    df = ProcessingData().get_complete_text(df=df)

    cm = ClusteringModerator()
    cm.bestfit_kmeans(df=df)


if __name__ == '__main__':
    if(_cluster_number!='None' or _cluster_number!=None): #If the cluster number given in cmd is None it goes for the percentage
        logger.info('Cluster number entered is {}'.format(str(_cluster_number)))
        initiate_kmeans(k_number=_cluster_number)

    else:
        logger.info('Cluster number entered is None')
        initiate_kmeans()


    # initiate_bestfit()
    # initiate_hierarchy(hc_number=6)
    # initiate_hierarchical_dendrogram()
    """
    pcm = PreClusteringModerator()
    df_to_csv = DataframeToCSV()
    CSV_Df = CsvToDataFrame()
    df, unique_values_dict=pcm.fetching_df()A

    d = {'horizontal': ['Fintech'], 'technology_segment_1': ['Digital Banking']}
    df = pcm.df_filtering(df=df,filter_dict=d)
    df = ProcessingData().get_text_without_stem(df=df)

    cm = ClusteringModerator()
    dict_of_cluster_and_tags, dict_of_cluster_and_filename=cm.kmeans_cluster_by_tfidf_vectorizer(df=df)

    # cm.bestfit_kmeans(df=df)

    dict_tag_df = df_to_csv.dict_to_df(dict_of_cluster_and_tags)
    df_to_csv.load_df_to_csv(df=dict_tag_df,cluster_csv="cluster_tag")

    dict_filename_df = df_to_csv.dict_to_df(dict_of_cluster_and_filename)
    df_to_csv.load_df_to_csv(df=dict_filename_df, cluster_csv="cluster_filename")
    
    """


