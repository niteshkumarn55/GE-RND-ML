#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 17:29:19 2018

@author: nitesh
"""

from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
import logging
import os
import math
from LoggingDetails.LogFileHandler import logs
from LoggingDetails.LogPathConstant import LogFilePathContants,LogFiles

log_file = os.path.join(LogFilePathContants._BASE_LOG_FILE,
                        LogFiles._CLUSTERING_LOG_FILE)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logs().fileHandler(logging,log_file_path=log_file)
logger.addHandler(file_handler)
class BestFitClusterTechnique():
    """
    This is used to find the Best fit of the number of cluster based on the real data and this number can be used on production code.
    """
    def cluster_by_elbow_method_for_mini_kmeans(self, max_cluster_iter=2,X_train_matrix=None):
        wcss = [] #within-cluster sums of squares
        if X_train_matrix != None:
            for i in range (1,max_cluster_iter):
                km = MiniBatchKMeans(n_clusters=i, init='k-means++', max_iter=100, n_init=10,
                                     init_size=1000, batch_size=1000, verbose=True, random_state=42)
                km.fit(X_train_matrix)
                wcss.append(km.inertia_)
            plt.plot(range(1, max_cluster_iter), wcss)
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            plt.show()

        else:
            print("Please Provide the X_train_matrix to fit into the mini batch kmeans")

    def cluster_by_elbow_method_for_kmeans(self, max_cluster_iter=2,X_train_matrix=None):
        wcss = [] #within-cluster sums of squares
        if X_train_matrix != None:
            for i in range (1,max_cluster_iter):
                km = KMeans(n_clusters=i, init='k-means++', max_iter=100, n_init=10,
                            verbose=True)
                km.fit(X_train_matrix)
                wcss.append(km.inertia_)
            plt.plot(range(1, max_cluster_iter), wcss)
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            plt.show()

        else:
            print("Please Provide the X_train_matrix to fit into the mini batch kmeans")

class RuleBasedBestFitClusters():

    def best_fit_clusters_based_df_count(self,count_of_categories=None, df_count=None,percentage_for_cluster_number=60):
        """

        :param count_of_categories:
        :param df_count:
        :param percentage_for_cluster_number:
        :return:
        """
        if (df_count != None or count_of_categories != None):
            try:
                while True:
                    cluster_number = math.ceil((count_of_categories / 100) * percentage_for_cluster_number)
                    temp_perc_of_clusters = ((cluster_number / df_count) *100)
                    if(temp_perc_of_clusters<1.5):
                        percentage_for_cluster_number = percentage_for_cluster_number + 10
                        logger.info("the bestfit percentage of cluster %s is more, ending up with less number"
                                    " clusters %s for data %s hence reducing the cluster number percentage",
                                    str(percentage_for_cluster_number),str(cluster_number),str(df_count))
                    elif(temp_perc_of_clusters>3):
                        percentage_for_cluster_number = percentage_for_cluster_number - 10
                        logger.info("the bestfit percentage for the cluster %s is very less, ending up with more"
                                    " number of cluster %s for data %s, hence increasing the cluster number percentage",
                                    str(percentage_for_cluster_number),str(cluster_number),str(df_count))
                    else:
                        logger.info("bestfit percenatage of the cluster is %s and the total number of cluster is %s "
                                    "for the records %s",str(percentage_for_cluster_number),str(cluster_number)
                                    ,str(df_count))
                        break
                return cluster_number
            except Exception as error:
                logger.error("Some issue in finding the best fit calc {}".format(str(error)))
        else:
            logger.warning("df count is empty can't find the best fit cluster or the count of categories is empty")
