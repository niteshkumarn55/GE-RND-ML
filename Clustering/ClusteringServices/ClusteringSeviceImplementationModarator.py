#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 16:27:19 2018

@author: nitesh
"""
import os
import sys
_PROJECT_EXECUTION_BASEPATH = r"/Users/nitesh/OneDrive/Work/GE_Python_Workspace/Clustering/"
print(os.path.abspath(os.path.join(os.getcwd(),os.pardir)))
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append(_PROJECT_EXECUTION_BASEPATH)

_cluster_number = None
_cluster_stem = None
_percentage_of_cluster_bestfit = None


import logging
from ClusteringServices.Utilities.ServiceUtilities import ExtraTools
from LoggingDetails.LogFileHandler import logs
from LoggingDetails.LogPathConstant import LogFilePathContants, LogFiles
from CSVUtilities.CSVToDf import CsvToDataFrame
from CSVUtilities.CSVAndDFToDict import CSVToDictionaryMapping
from CSVUtilities.DFToCSV import DataframeToCSV
from DataMassaging.DataPreprocessingUtility.Utilities import DataUtilities, DFAnalyse
from Vectorizers.TfidfVectorizing import VectorizingTechnique as TfidfVectorizingTechnique
from Clusters.KmeansClustering.KmeansClusteringProcess import KmeansTechnique
from Clusters.AffinityClustering.AffinityPropagationProcess import AffinityPropagationTechnique
from DataMassaging.DataPreprocessingUtility.Preprocessing.ContentProcessing import ProcessingData
from Clusters.BestFitCluster.BestFitNumberOfCluster import BestFitClusterTechnique, RuleBasedBestFitClusters
from Clusters.HierarchicalClustering.HierarchicalClusteringProcess import HierarchyTechnique
from DistanceMatrix.DistanceMatrixOfDocuments import EuclideanDistanceForDocuments
from DBO.ClusteringDBOImpl import ClusterCurd
from DBO.DataBaseServices import DBConnection
import pandas as pd
import os


log_file = os.path.join(LogFilePathContants._BASE_LOG_FILE,
                        LogFiles._CLUSTERING_LOG_FILE)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logs().fileHandler(logging, log_file_path=log_file)
logger.addHandler(file_handler)

try:
    _job_id = 'None'
    if len(sys.argv) == 2:
        _cluster_preference = _cluster_number = sys.argv[1]

        if(_cluster_number != 'None' or _cluster_preference != None):
            _cluster_number = int(_cluster_number)
            _cluster_preference = float('%.1f' % float(sys.argv[1]))

        print('the cluster number is: ', _cluster_number,
              ' and the type is ', type(_cluster_number))

    elif len(sys.argv) == 3:

        _cluster_preference = _cluster_number = sys.argv[1]
        _cluster_stem = str(sys.argv[2])

        if (_cluster_number != 'None'):
            _cluster_number = int(_cluster_number)
            _cluster_preference = float('%.1f' % float(_cluster_preference))

        print('The cluster number is: ', _cluster_number,
              ' and the type is ', type(_cluster_number))
        print('the cluster stem or not:', _cluster_stem,
              ' and the type is ', type(_cluster_stem))

    elif len(sys.argv) == 4:
        _cluster_preference = _cluster_number = sys.argv[1]
        _cluster_stem = str(sys.argv[2])
        _percentage_of_cluster_bestfit = sys.argv[3]

        if (_cluster_number != 'None'):
            _cluster_number = int(_cluster_number)
            _cluster_preference = float('%.1f' % float(sys.argv[1]))

        if(_percentage_of_cluster_bestfit != 'None'):
            _percentage_of_cluster_bestfit = int(
                _percentage_of_cluster_bestfit)

        print('The cluster number is: ', _cluster_number,
              ' and the type is ', type(_cluster_number))
        print('the cluster stem or not:', _cluster_stem,
              ' and the type is ', type(_cluster_stem))
        print('the percentage is: ', _percentage_of_cluster_bestfit,
              ' and the type is ', type(_percentage_of_cluster_bestfit))

    elif len(sys.argv) == 5:
        _cluster_preference = _cluster_number = sys.argv[1]
        _cluster_stem = str(sys.argv[2])
        _percentage_of_cluster_bestfit = sys.argv[3]
        _algo_name = sys.argv[4]

        if (_cluster_number != 'None'):
            _cluster_number = int(_cluster_number)
            _cluster_preference = float('%.1f' % float(sys.argv[1]))

        if (_percentage_of_cluster_bestfit != 'None'):
            _percentage_of_cluster_bestfit = int(
                _percentage_of_cluster_bestfit)

        if(_algo_name == 'None' or _algo_name == None):
            _algo_name = 'kmeans'

        print('The cluster number is: ', _cluster_number,
              ' and the type is ', type(_cluster_number))
        print('the cluster stem or not:', _cluster_stem,
              ' and the type is ', type(_cluster_stem))
        print('the percentage is: ', _percentage_of_cluster_bestfit,
              ' and the type is ', type(_percentage_of_cluster_bestfit))
        print('The algo used is: ', _algo_name,
              ' and the type is: ', type(_algo_name))
    elif len(sys.argv) == 6:
        _cluster_preference = _cluster_number = sys.argv[1]
        _cluster_stem = str(sys.argv[2])
        _percentage_of_cluster_bestfit = sys.argv[3]
        _algo_name = sys.argv[4]
        _job_id = sys.argv[5]

        if (_cluster_number != 'None'):
            _cluster_number = int(_cluster_number)
            _cluster_preference = float('%.1f' % float(sys.argv[1]))

        if (_percentage_of_cluster_bestfit != 'None'):
            _percentage_of_cluster_bestfit = int(
                _percentage_of_cluster_bestfit)

        if(_algo_name == 'None' or _algo_name == None):
            _algo_name = 'kmeans'

        print('The cluster number is: ', _cluster_number,
              ' and the type is ', type(_cluster_number))
        print('the cluster stem or not:', _cluster_stem,
              ' and the type is ', type(_cluster_stem))
        print('the percentage is: ', _percentage_of_cluster_bestfit,
              ' and the type is ', type(_percentage_of_cluster_bestfit))
        print('The algo used is: ', _algo_name,
              ' and the type is: ', type(_algo_name))
        print('The algo used is: ', _job_id,
              ' and the type is: ', type(_job_id))

except Exception as error:
    logger.error(
        "Some issue taking the argument from the terminal {}".format(str(error)))


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
                    if(_job_id!=None):
                        df = CSV_Df.get_df_from_csv(job_id=_job_id)
                    else:
                        df = CSV_Df.get_df_from_csv(job_id=None)
                    logger.info("CSV Loaded to the dataframe")
                    df_independent_fields = DataUtilities(df=df)

                    try:
                        if df_independent_fields == None:
                            raise Exception
                        else:
                            df, independent_fields = df_independent_fields.show_hori_and_tech_segments_from_df()
                            logger.info(
                                "removed the NaN from auto recognized independent variables")
                            logger.info("Populating the recognized independent fields {}".format(str(independent_fields
                                                                                                     )))
                            df_independent_unique_values = DFAnalyse()
                            try:
                                if df_independent_unique_values == None:
                                    raise Exception
                                else:
                                    unique_values_dict = df_independent_unique_values.\
                                        get_unique_fields_based_on_tech_and_horizontal(df=df,
                                                                                       independent_fields=independent_fields)
                                    logger.info("The unique values for the independent fields are {}"
                                                .format(str(unique_values_dict)))

                            except Exception as error:
                                logger.error("DF Analyse failed to provide the unique values {}".format(str(error)),
                                             exc_info=True)

                    except Exception as error:
                        logger.error("Data Utility filtering failed {}".format(
                            str(error)), exc_info=True)

            except Exception as error:
                logger.error("CSV not loaded {}".format(
                    str(error)), exc_info=True)

            return df, unique_values_dict
        except Exception as error:
            logger.error("Some Issue in fetching the data frame {}".format(
                str(error)), exc_info=True)

    def df_filtering(self, df=pd.DataFrame(), filter_dict=dict()):
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
            if df_filter_category == None and len(df) <= 0 and len(filter_dict) <= 0:
                raise Exception
            else:
                df = df_filter_category.filter_df_by_category(
                    df=df, dict_category_and_values=filter_dict)
                if not len(df) > 0:
                    raise Exception

            return df
        except Exception as error:
            logger.error("df filtering of category failed {}".format(
                str(error)), exc_info=True)


class ClusteringModerator():

    def kmeans_cluster_by_tfidf_vectorizer(self, df=pd.DataFrame(), cluster_number=1):
        """

        :param df:
        :return:
        """

        # Tfidf vectorization process to get the vectors for bow for all documents with in the doctype
        try:
            vector_bow = TfidfVectorizingTechnique()
            """kmeans_Clustering is performed without tfidf pipelined in the count vectorizing matrix"""
            contents = df['processed_text'].tolist()
            # company_id = df['ID'].tolist()
            company_id = df['domain_name'].tolist()
            X_train_vecs, vectorizer = vector_bow.tfidf_vectorizer(contents)
            logger.info(
                "Preprocessed data got converted into features -> vectors")
            kmeans = KmeansTechnique()
            # Kmeans without the LSA dimensionality reduction
            cluster_and_terms_map, cluster_and_count_of_docs, dict_of_cluster_and_tags, dict_of_cluster_and_filename = \
                kmeans.kmeans_cluster_technique(number_cluster=cluster_number,
                                                X_train_vecs=X_train_vecs, vectorizer=vectorizer, filenames=company_id, contents=contents,
                                                is_dimension_reduced=False)

            # TODO: See how to enble the Vectorizer.get_feature_names() for the pipelined tfidf vectors.
            # TODO: Follow the commented steps in the mini_kmeans_cluster_by_count_vectorizer() above, after figuring out how to get the get_feature_names()
            print("dict_of_cluster_and_tags:", dict_of_cluster_and_tags)
            print("dict_of_cluster_and_filename:",
                  dict_of_cluster_and_filename)
            return dict_of_cluster_and_tags, dict_of_cluster_and_filename
        except Exception as error:
            logger.error("Kmeans clustering failed {}".format(str(error)))

    def hierachical_dendogram_bestfit(self, df=pd.DataFrame()):
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
            logger.info(
                "Preprocessed data got converted into features for dendogram-> vectors")
            dendrogram = HierarchyTechnique()
            dendrogram.hierarchy_dendrogram(
                X_train_vecs=X_train_vecs, filenames=company_id)
            logger.info("Hierarchy dendrogram is created")
        except Exception as error:
            logger.error("Dendrogram failed {}".format(str(error)))

    def hierarchical_cluster_by_tfidf_vectorizer(self, df=pd.DataFrame(), cluster_number=1):
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
            logger.info(
                "Preprocessed data got converted into features -> vectors")
            hierarchy = HierarchyTechnique()
            cluster_and_terms_map, cluster_and_count_of_docs, dict_of_cluster_and_tags, dict_of_cluster_and_filename = \
                hierarchy.hierarchy_cluster_technique(number_cluster=cluster_number,
                                                      X_train_vecs=X_train_vecs, vectorizer=vectorizer, filenames=company_id,
                                                      contents=contents,
                                                      is_dimension_reduced=False)

            print("dict_of_cluster_and_tags:", dict_of_cluster_and_tags)
            print("dict_of_cluster_and_filename:",
                  dict_of_cluster_and_filename)
            return dict_of_cluster_and_tags, dict_of_cluster_and_filename
        except Exception as error:
            logger.error(
                "Hierarchical clustering failed {}".format(str(error)))

    def affinity_propagation_by_tfidf_vectorizer(self, df=pd.DataFrame(), cluster_preferences=None):

        try:
            vector_bow = TfidfVectorizingTechnique()
            contents = df['processed_text'].tolist()
            company_id = df['domain_name'].tolist()
            X_train_vecs, vectorizer = vector_bow.tfidf_vectorizer(contents)
            logger.info(
                "Preprocessed data got converted into features -> vectors")
            affinity = AffinityPropagationTechnique()
            ed = EuclideanDistanceForDocuments()
            dict_of_cluster_and_filename, X, y, exemplars = affinity.affinity_cluster_technique(
                preference=cluster_preferences, X_train_vecs=X_train_vecs, filenames=company_id, contents=contents)
            # centeriod_and_data_radius = ed.euclidean_distance_matrix(X=X,y=y,exemplars=exemplars)
            centeriod_and_data_radius = None

            return dict_of_cluster_and_filename, centeriod_and_data_radius
        except Exception as error:
            logger.error("Affinity Propagation fialed {}".format(str(error)))

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
            logger.info(
                "Preprocessed data got converted into features for bestfit-> vectors")
            bestfit_kmeans = BestFitClusterTechnique()
            bestfit_kmeans.cluster_by_elbow_method_for_kmeans(
                max_cluster_iter=300, X_train_matrix=X_train_vecs)
            logger.info("Elbow method is complete and the graph is populated")
        except Exception as error:
            logger.error("Best fit of kmeans failed {}".format(str(error)))


def initiate_kmeans(k_number=None):
    """

    :param k_number:
    :return:
    """
    pcm = PreClusteringModerator()
    bfit = RuleBasedBestFitClusters()
    util = ExtraTools()
    df_to_csv = DataframeToCSV()
    csv_and_dict = CSVToDictionaryMapping()

    df, unique_values_dict = pcm.fetching_df()
    filtered_tech_dict = util.get_filtered_dict()

    df = pcm.df_filtering(df=df, filter_dict=filtered_tech_dict)

    count_of_categories = util.get_number_of_categories(
        unique_values_dict_categories=unique_values_dict)

    # Dataframe after the preprocessing --------------------
    if(_cluster_stem == 'stem'):  # Steming happens only when you ask it from cmd prompt
        logger.info('The process is stem from the cmd')
        df = ProcessingData().get_complete_text(df=df)  # stemming the text

    else:
        logger.info('The process is not stemed')
        df = ProcessingData().get_text_without_stem(
            df=df)  # No stemming is done for the text

    if(k_number == None):
        # If the cluster number is None then it takes a default percentage which is 60%
        if(_percentage_of_cluster_bestfit == 'None' or _percentage_of_cluster_bestfit == None):
            logger.info('The lenght if the df is {}'.format(str(len(df))))
            logger.info('Default percentage is consider which is 60%')
            k_number = bfit.best_fit_clusters_based_df_count(
                count_of_categories=count_of_categories, df_count=len(df))

        else:  # If Percentage is given in the cmd then percentage is override from the default percentage which is 60%
            logger.info('The lenght if the df is {}'.format(str(len(df))))
            logger.info('Default percentage is overridden for the percentage given {}'
                        .format(str(_percentage_of_cluster_bestfit)))
            k_number = bfit.best_fit_clusters_based_df_count(count_of_categories=count_of_categories, df_count=len(df),
                                                             percentage_for_cluster_number=_percentage_of_cluster_bestfit)

    logger.info("The total number of cluster is %s for the records %s",
                str(k_number), str(len(df)))
    cm = ClusteringModerator()
    dict_of_cluster_and_tags, dict_of_cluster_and_filename = cm.kmeans_cluster_by_tfidf_vectorizer(df=df,
                                                                                                   cluster_number=k_number)

    # cm.bestfit_kmeans(df=df)

    if(_job_id=='None' or _job_id==None):
        dict_tag_df = csv_and_dict.dict_to_df(dict_of_cluster_and_tags)
        df_to_csv.load_df_to_csv(
            df=dict_tag_df, cluster_csv="cluster_tag", algo="kmeans")

        dict_filename_df = csv_and_dict.dict_to_df(dict_of_cluster_and_filename)
        df_to_csv.load_df_to_csv(
            df=dict_filename_df, cluster_csv="cluster_filename", algo="kmeans")
    else:
        # Save the data to DB
        logger.info("The kmeans job id is :{}".format(str(_job_id)))
        clusterDBO = ClusterCurd()
        dict_filename_df = csv_and_dict.dict_to_df(dict_of_cluster_and_filename)

        result = clusterDBO.get_job_tbl_by_jobid(_job_id)
        job = None
        for row in result:
            job = row['job_id']

        # Save job_id and cluster_name
        df_jobid_clustername = pd.DataFrame()
        df_jobid_clustername['cluster_name'] = dict_filename_df.index
        if job != None: df_jobid_clustername['job_id'] = job
        logger.info(
            "Inserting for job_cluster_mapper with data.... the job_id {} and cluster_name {}  ".format(str(job), str(
                dict_filename_df.index)))
        clusterDBO.insert_job_id_and_cluster_name(
            df=df_jobid_clustername)  # Inserting into job_id and clustername table

        # Save Cluster name and its respective filenames
        try:
            cluster_result = clusterDBO.get_cluster_id_by_jc_id(job) if (job != None) else None
            if cluster_result != None:
                for cluster_row in cluster_result:
                    jc_id = cluster_row['jc_id']
                    cluster_name = cluster_row['cluster_name']

                    df_cluster_company_name = pd.DataFrame()
                    df_cluster_company_name['company_id'] = dict_of_cluster_and_filename[cluster_name]
                    df_cluster_company_name['jc_id'] = jc_id

                    logger.info("df formed and sending for insertion operation for the cluster from kmeans {} and the jc_id is {}".format(str(cluster_name),str(jc_id)))
                    clusterDBO.insert_jc_id_and_filename(df=df_cluster_company_name)
            else:
                logger.error("Result is empty from db from the tbl cluster and job id mapping where job_id is {}".format(str(job)))
                raise Exception

            try:
                clusterDBO.update_status_in_job_table(status='Done', job_id=job)
                logger.info("Job details table updated on status")
            except Exception as e:
                logger.error(
                    "Something went wrong in updating the job_table for the completed status {}".format(str(e)))

            logger.info("Kmeans cluster and filename/domain name DB operation is succesfully completed")

        except Exception as e:
            logger.error("Some error in saving the cluster name and its respective filename from kmeans {}".format(str(e)))
            logger.error("Kmeans clusters DB save Failed")


def initiate_affinity_propagation(cluster_preferences=None):
    pcm = PreClusteringModerator()
    bfit = RuleBasedBestFitClusters()
    util = ExtraTools()
    df_to_csv = DataframeToCSV()
    csv_and_dict = CSVToDictionaryMapping()
    df, unique_values_dict = pcm.fetching_df()
    filtered_tech_dict = util.get_filtered_dict()

    df = pcm.df_filtering(df=df, filter_dict=filtered_tech_dict)

    count_of_categories = util.get_number_of_categories(
        unique_values_dict_categories=unique_values_dict)

    # Dataframe after the preprocessing --------------------
    if (_cluster_stem == 'stem'):  # Steming happens only when you ask it from cmd prompt
        logger.info('The process is stem from the cmd')
        df = ProcessingData().get_complete_text(df=df)  # stemming the text

    else:
        logger.info('The process is not stemed')
        df = ProcessingData().get_text_without_stem(
            df=df)  # No stemming is done for the text

    cm = ClusteringModerator()
    dict_of_cluster_and_filename, centeriod_and_data_radius = cm.affinity_propagation_by_tfidf_vectorizer(
        df=df, cluster_preferences=cluster_preferences)

    if (_job_id == 'None' or _job_id == None):
        dict_filename_df = csv_and_dict.dict_to_df(dict_of_cluster_and_filename)
        df_to_csv.load_df_to_csv(df=dict_filename_df, cluster_csv="cluster_filename", algo="affinity_propagation")

        # dict_radius_df = csv_and_dict.dict_to_df(centeriod_and_data_radius)
        # df_to_csv.load_df_to_csv(df=dict_radius_df, cluster_csv="cluster_distance", algo="affinity_propagation")
    else:

        # Save the data to DB
        logger.info("The affinity job id is :{}".format(str(_job_id)))
        clusterDBO = ClusterCurd()
        dict_filename_df = csv_and_dict.dict_to_df(dict_of_cluster_and_filename)

        result = clusterDBO.get_job_tbl_by_jobid(_job_id)
        job = None
        for row in result:
            job = row['job_id']

        # Save job_id and cluster_name
        df_jobid_clustername = pd.DataFrame()
        df_jobid_clustername['cluster_name'] = dict_filename_df.index
        if job!=None: df_jobid_clustername['job_id'] = job
        logger.info("Inserting for job_cluster_mapper with data.... the job_id {} and cluster_name {}  ".format(str(job), str(
            dict_filename_df.index)))
        clusterDBO.insert_job_id_and_cluster_name(df=df_jobid_clustername) #Inserting into job_id and clustername table


        # Save Cluster name and its respective filenames
        try:
            cluster_result = clusterDBO.get_cluster_id_by_jc_id(job) if (job!=None) else None
            if cluster_result != None:
                for cluster_row in cluster_result:
                    jc_id = cluster_row['jc_id']
                    cluster_name = cluster_row['cluster_name']

                    df_cluster_company_name = pd.DataFrame()
                    df_cluster_company_name['company_id'] = dict_of_cluster_and_filename[cluster_name]
                    df_cluster_company_name['jc_id'] = jc_id

                    logger.info(
                        "df formed and sending for insertion operation for the cluster from affinity {} and the jc_id "
                        "is {}".format(
                            str(cluster_name), str(jc_id)))
                    clusterDBO.insert_jc_id_and_filename(df=df_cluster_company_name)
            else:
                logger.error(
                    "Result is empty from db from the tbl cluster and job id mapping where job_id is {}".format(
                        str(job)))
                raise Exception

            try:
                clusterDBO.update_status_in_job_table(status='Done', job_id=job)
            except:
                logger.error("Something went wrong in updating the job_table for the completed status")

            logger.info("Affinity cluster and filename/domain name DB operation is succesfully completed")

        except Exception as e:
            logger.error("Some error in saving the cluster name and its respective filename from affinity {}"
                         .format(str(e)))
            logger.error("Affinity clusters DB save Failed")




    # df_to_sql = DFToSQl()
    # df_to_sql.save_df_to_sql(dict_filename_df)


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
    df_to_csv.load_df_to_csv(
        df=dict_tag_df, cluster_csv="cluster_tag", algo="hierarchy")

    dict_filename_df = csv_and_dict.dict_to_df(dict_of_cluster_and_filename)
    df_to_csv.load_df_to_csv(
        df=dict_filename_df, cluster_csv="cluster_filename", algo="hierarchy")


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
    # _algo_name = 'kmeans'
    # _cluster_preference = 'None'
    # _job_id = '20180320220207'
    # _cluster_number = 5
    if(_algo_name == 'kmeans'):
        # If the cluster number given in cmd is None it goes for the percentage
        if(_cluster_number != 'None' or _cluster_number != None):
            logger.info('Kmeans Cluster number entered is {}'.format(
                str(_cluster_number)))
            try:
                initiate_kmeans(k_number=_cluster_number)
            except Exception as e:
                logger.error("Failure occurred in kmeans {}".format(str(e)))

        else:
            logger.info('Cluster number entered is None')
            try:
                initiate_kmeans()
            except Exception as e:
                logger.error("Failure occurred in kmeans {}".format(str(e)))

    elif(_algo_name == 'affinity'):
        if(_cluster_preference != 'None' or _cluster_preference != None):
            logger.info('Affinity Cluster entered and the preference is {}'.format(
                str(_cluster_preference)))
            try:
                initiate_affinity_propagation(
                    cluster_preferences=_cluster_preference)
            except Exception as e:
                logger.error("Failure occurred in affinity {}".format(str(e)))
        else:
            logger.info('Affinity cluster with no preferences set... started')
            try:
                initiate_affinity_propagation()
            except Exception as e:
                logger.error("Failure occurred in affinity {}".format(str(e)))

    # initiate_bestfit()
    # initiate_hierarchy(hc_number=6)
    # initiate_hierarchical_dendrogram()
