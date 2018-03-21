#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 17:25:19 2018

@author: nitesh
"""

from sklearn.cluster import KMeans
from time import time
import pandas as pd
from LoggingDetails.LogFileHandler import logs
import logging
import os
from LoggingDetails.LogPathConstant import LogFilePathContants,LogFiles
log_file = os.path.join(LogFilePathContants._BASE_LOG_FILE,
                        LogFiles._CLUSTERING_LOG_FILE)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logs().fileHandler(logging,log_file_path=log_file)
logger.addHandler(file_handler)

class KmeansTechnique():

    def kmeans_cluster_technique(self, number_cluster, X_train_vecs, vectorizer, filenames, contents, svd=None, is_dimension_reduced=True):
        try:
            logger.info("Inside the kmeans clustering engine")
            km = KMeans(n_clusters=number_cluster, init='k-means++', max_iter=100, n_init=10,
                        verbose=True)
            logger.info("Clustering sparse data with {}".format(str(km)))
            # print("Clustering sparse data with %s" % km)
            t0 = time()
            km.fit(X_train_vecs)
            logger.info("done in  {}".format(str(time() - t0)))
            # print()
            cluster_labels = km.labels_.tolist()
            logger.info("List of cluster names is :{}".format(str(len(cluster_labels))))
            data = {'filename':filenames, 'contents':contents, 'cluster_label':cluster_labels}
            frame = pd.DataFrame(data=data, index=[cluster_labels], columns=['filename', 'contents', 'cluster_label'])
            cluster_and_count_of_docs = frame['cluster_label'].value_counts(sort=True, ascending=False)
            logger.info("The cluster and count of docs dict size is : ".format(len(cluster_and_count_of_docs)))
            # print()
            grouped = frame['cluster_label'].groupby(frame['cluster_label'])
            logger.info("the cluster frame grouped mean my cluster label is :".format(str(grouped.mean())))
            # print()
            logger.info("Finding the top terms per cluster------")
            logger.info("Top Terms Per Cluster will be generated-------")

            if is_dimension_reduced:
                if svd != None:
                    original_space_centroids = svd.inverse_transform(km.cluster_centers_)
                    order_centroids = original_space_centroids.argsort()[:, ::-1]
            else:
                order_centroids = km.cluster_centers_.argsort()[:, ::-1]

            terms = vectorizer.get_feature_names()
            cluster_and_terms_map = {}# Used to get the map of terms for each cluster and eventually is stored in a file for further analysis.
            dict_of_cluster_and_tags = dict()
            dict_of_cluster_and_filename = dict()
            for i in range(number_cluster):
                logger.info("Cluster {}:".format(str(i)))
                list_of_cluster_tags = list()
                for ind in order_centroids[i, :15]:
                    # print(' %s' % terms[ind], end=',')
                    list_of_cluster_tags.append(terms[ind])

                # print()
                logger.info("Cluster and the filename is generated...")
                # print("Cluster %d filenames:" % i, end='')
                list_of_cluster_filename = list()
                for file in frame.ix[i]['filename'].values.tolist():
                    # print(' %s,' % file, end='')
                    list_of_cluster_filename.append(str(file))
                # print()
                logger.info("generating the dictionary of cluster tags and cluster filename")
                dict_of_cluster_and_tags['cluster '+str(i)] = list_of_cluster_tags
                dict_of_cluster_and_filename['cluster '+str(i)] = list_of_cluster_filename


            #you can comment if you don't what terms to be appended in a text file for analysis purpose
                """To check the terms and what kind of terms are choosen by the cluster we load it to a plain text file"""
                list_of_terms = []  # Used to load the list of terms from each cluster to a file, used for further analysis purpose only.
                for ind in order_centroids[i, :15]:
                    list_of_terms.append(terms[ind])
                cluster_and_terms_map[i] = list_of_terms

            logger.info("Clustering is finished send the data to process and load it for the csv or the sql")
            return cluster_and_terms_map,cluster_and_count_of_docs,dict_of_cluster_and_tags,dict_of_cluster_and_filename #Returning the cluster and terms of each cluster. This is eventually used for analysis, but nothing to do the algorithm.
        except Exception as e:
            logger.info("Some error occured in kmeans clustering and failed to cluster {}".format(str(e)))
            return None, None, None, None





