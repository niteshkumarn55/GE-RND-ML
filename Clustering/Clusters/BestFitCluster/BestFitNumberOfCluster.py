#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 17:29:19 2018

@author: nitesh
"""

from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
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
                            verbose=True, random_state=42)
                km.fit(X_train_matrix)
                wcss.append(km.inertia_)
            plt.plot(range(1, max_cluster_iter), wcss)
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            plt.show()

        else:
            print("Please Provide the X_train_matrix to fit into the mini batch kmeans")
