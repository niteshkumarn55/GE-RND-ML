import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import re
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.probability import FreqDist
from collections import defaultdict
from nltk.corpus import stopwords
from string import punctuation
from heapq import nlargest
import math
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import time
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


class AffinityPropagationTechnique():

    def affinity_cluster_technique(self, preference=None, X_train_vecs=None, filenames=None, contents=None):
        """

        :param preference:
        :param X_train_vecs:
        :param vectorizer:
        :param filenames:
        :param contents:
        :return:
        """
        logger.info('Into the affinity core engine having the preference {}'.format(str(preference)))
        if X_train_vecs!=None or X_train_vecs!='None':
            X = X_train_vecs
            # X = cosine_distances(X)



            # svd = TruncatedSVD(n_components=100)
            # normalizer = Normalizer(copy=False)
            # lsa = make_pipeline(svd, normalizer)
            # X= X_train_vecs = lsa.fit_transform(X_train_vecs)

            # X = StandardScaler().fit_transform(X)
            logger.info("The shape of X_train after the lsa {}".format(X_train_vecs.shape))
            # X = X_train_vecs.toarray()
            # X = np.array(X)
            logger.info('Vector to array of the X data processed')
            if preference!='None':
                af = AffinityPropagation(damping=0.5, preference=preference,verbose=True)
            else:
                af = AffinityPropagation(damping=0.5, preference=None,verbose=True)
            logger.info('The affinity propagation object is {}'.format(str(af)))
            y = af.fit_predict(X)
            exemplars = af.cluster_centers_
            number_of_clusters = af.labels_.tolist()

            # logger.info("The total number of cluster Affinity generated is: {}".format(str(len(exemplars))))
            data = {'filename': filenames, 'contents': contents, 'cluster_label': number_of_clusters}
            frame = pd.DataFrame(data=data, index=[number_of_clusters], columns=['filename', 'contents', 'cluster_label'])
            logger.info('Sample of the clustered df {}'.format(str(frame.head(2))))
            cluster_and_count_of_docs = frame['cluster_label'].value_counts(sort=True, ascending=False)
            dict_of_cluster_and_filename = dict()

            for i in number_of_clusters:
                list_of_file_id = list()
                list_of_files_in_cluster = list()
                list_of_files_in_cluster = frame.ix[i]['filename'].tolist()
                try:
                    for file_id in list_of_files_in_cluster:
                        list_of_file_id.append(file_id)
                    dict_of_cluster_and_filename['clusters ' + str(i)] = list_of_file_id
                except Exception as e:
                    list_of_file_id.append(list_of_files_in_cluster)
                    dict_of_cluster_and_filename['clusters ' + str(i)] = list_of_file_id

            return dict_of_cluster_and_filename, X, y, exemplars
        else:
            try:
                raise Exception
            except Exception as e:
                logger.error("The X_train data is None {}".format(str(e)))
            return dict(),None,None,None