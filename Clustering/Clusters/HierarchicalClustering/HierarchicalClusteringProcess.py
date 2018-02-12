#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 04:28:19 2018

@author: nitesh
"""
from scipy.cluster.hierarchy import ward,dendrogram
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import pandas as pd

class HierarchyTechnique():

    def hierarchy_cluster_technique(self, number_cluster, X_train_vecs, vectorizer, filenames, contents, svd=None, is_dimension_reduced=True):
        """

        :param number_cluster:
        :param X_train_vecs:
        :param vectorizer:
        :param filenames:
        :param contents:
        :param svd:
        :param is_dimension_reduced:
        :return:
        """
        hc = AgglomerativeClustering(n_clusters=number_cluster, affinity="euclidean", linkage='ward')

        hc = hc.fit_predict(X_train_vecs)
        cluster_labels = hc.labels_.tolist()
        print("List of the cluster names is : ", cluster_labels)
        data = {'filename': filenames, 'contents': contents, 'cluster_label': cluster_labels}
        frame = pd.DataFrame(data=data, index=[cluster_labels], columns=['filename', 'contents', 'cluster_label'])
        cluster_and_count_of_docs = frame['cluster_label'].value_counts(sort=True, ascending=False)
        print(cluster_and_count_of_docs)
        print()
        print(cluster_and_count_of_docs)
        print()
        grouped = frame['cluster_label'].groupby(frame['cluster_label'])
        print(grouped.mean())
        print()
        print("Top Terms Per Cluster :")

        if is_dimension_reduced:
            if svd != None:
                original_space_centroids = svd.inverse_transform(hc.cluster_centers_)
                order_centroids = original_space_centroids.argsort()[:, ::-1]
        else:
            order_centroids = hc.cluster_centers_.argsort()[:, ::-1]

        terms = vectorizer.get_feature_names()
        cluster_and_terms_map = {}  # Used to get the map of terms for each cluster and eventually is stored in a file for further analysis.
        dict_of_cluster_and_tags = dict()
        dict_of_cluster_and_filename = dict()
        for i in range(number_cluster):
            print("Cluster %d:" % i, end=' ')
            list_of_cluster_tags = list()
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind], end=',')
                list_of_cluster_tags.append(terms[ind])

            print()
            print("Cluster %d filenames:" % i, end='')
            list_of_cluster_filename = list()
            for file in frame.ix[i]['filename'].values.tolist():
                print(' %s,' % file, end='')
                list_of_cluster_filename.append(int(file))
            print()
            dict_of_cluster_and_tags['cluster ' + str(i)] = list_of_cluster_tags
            dict_of_cluster_and_filename['cluster ' + str(i)] = list_of_cluster_filename

            # you can comment if you don't what terms to be appended in a text file for analysis purpose
            """To check the terms and what kind of terms are choosen by the cluster we load it to a plain text file"""
            list_of_terms = []  # Used to load the list of terms from each cluster to a file, used for further analysis purpose only.
            for ind in order_centroids[i, :10]:
                list_of_terms.append(terms[ind])
            cluster_and_terms_map[i] = list_of_terms

        return cluster_and_terms_map, cluster_and_count_of_docs, dict_of_cluster_and_tags, dict_of_cluster_and_filename  # Returning the cluster and terms of each cluster. This is eventually used for analysis, but nothing to do the algorithm.

def hierarchy_dendrogram(self, X_train_vecs, filenames):
        """

        :param X_train_vecs:
        :return:
        """
        dist = 1 - cosine_similarity(X_train_vecs)
        linkage_matrix = ward(dist)

        fig, ax = plt.subplots(figsize=(15, 20))  # set size
        ax = dendrogram(linkage_matrix, orientation="right", labels=filenames);

        plt.tick_params( \
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            labelbottom='off')

        plt.tight_layout()  # show plot with tight layout

        # uncomment below to save figure
        plt.savefig('ward_clusters.png', dpi=200)  # save figure as ward_clusters
        plt.close()


