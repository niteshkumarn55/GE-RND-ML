from flask import Flask,request
import json
import random
import logging
import os
import sys
from ClusteringServices.ClusteringSeviceImplementationModarator import PreClusteringModerator,ClusteringModerator
from DataMassaging.DataPreprocessingUtility.Preprocessing.ContentProcessing import ProcessingData
from os.path import dirname, abspath

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',level=logging.DEBUG)
parent_directory = dirname(dirname(abspath(__file__)))
logging.info("parent directory is {}".format(str(parent_directory)))
sys.path.append(parent_directory) #Adding to the sys path, so the modules sibling and under this can be seen

app = Flask(__name__)
print(app)
@app.route('/cluster/' , methods=['GET','POST'])
def Clustering():

    query = request.args.get('query')
    print(query)
    d = json.loads('UTF-8',query)
    pcm = PreClusteringModerator()
    df, unique_values_dict = pcm.fetching_df()

    # d = {'horizontal': ['Fintech'], 'technology_segment_1': ['Digital Banking']}
    df = pcm.df_filtering(df=df, filter_dict=d)
    df = ProcessingData().get_text_without_stem(df=df)

    cm = ClusteringModerator()
    cm.kmeans_cluster_by_tfidf_vectorizer(df=df)


if __name__ == "__main__":
    host = '0.0.0.0'
    app.run(host=host ,debug=True)

