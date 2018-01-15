#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 16:14:36 2017

@author: nitesh
"""
import sys
import os
where_am_i = os.getcwd()+'/' #Knowing the current directory from where the program is executing
sys.path.append(where_am_i) #Adding to the sys path, so the modules sibling and under this can be seen

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from Preprocessing.DataMassaging import DataPreprocessing
from Utilities.CustomCsv import CsvOperation
from Classifiers.MultinomialClassifer import MultinomialClassifierSegment
from Classifiers.Decisiontree import DecisionTreeSegment
from Classifiers.KNNClassifier import KnnClassifierSegment
from sklearn. ensemble import RandomForestClassifier
from Utilities.SplitDatasets import CustomSplits
from Ensemblers.SklearnEnsemblingClassifierMechanism import EnsemblingMechanism
from sklearn import svm
import pandas as pd
from Preprocessing.DataMassaging import DataPreprocessing
from MatplotCharts.Charts import GEPlots


# def get_complete_text(df=None,technology_column=None):
#     """
#     Takes the specialties, short_desc, about us and concatinates and provides a complete contacatated result in the df
#     :param df: Need the df which has the above mentioned columns
#     :return dataframe
#     """
#     df['complete_text'] = df[['specialties','specialties','short_description','about_us']].apply(lambda x: x.str.cat(sep=' '), axis=1)
#     df['processed_text'] = df['complete_text'].apply(DataPreprocessing().nltk_preprocessing)
# #    X = df.iloc['processed_text']
# #    Y = df.iloc[technology_column]
#     X = pd.DataFrame()
#     Y = pd.DataFrame()
#     X['processed_text'] = df['processed_text']
#     X['domain_name'] = df['domain_name']
#     Y[technology_column] = df[technology_column]
#     return df, X, Y

def training_and_modeling(df=None, X=None, Y=None, technology_column=None,model_path=None):
    """
    """
    split = CustomSplits()
    #tfidf vectorizer object to fit the documents and convert it in to numerical format
    doc_vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    doc_vectorizer = CountVectorizer()
    columns = ['classifier_name', 'vectorizer', 'model', 'prediction_accuracy']
    accuracy_df = pd.DataFrame()


    for model_no in range(10):
        model_name = 'model_'+str(model_no)
        print("*******************The i iteration is ", model_name)

        labels = Y[technology_column]
        X_train, X_test, Y_train, Y_test = split.random_X_and_Y(X=X,Y=labels)
        np_x_train = X_train['processed_text']
        np_x_test = X_test['processed_text']


        # X_train, X_test, Y_train, Y_test = split.shuffle_by_category(df=df,technology_column=technology_column)

        prediction_df = pd.DataFrame()
        prediction_df['domain_name'] = X_test['domain_name']
        prediction_df['Test_document'] = X_test['processed_text']
        prediction_df['Expected_result'] = Y_test



        # print("the dataframe length ", prediction_df.size)
        # index = []
        # for i in range(prediction_df.size):
        #     index.append(i)

        # prediction_df.set_index([index])
        # New test data from the training data, this is called as cross validation
        # cross_test_data = cross_validatation(trained_data=training_data,technology_column=technology_column)

        multi_classifier = MultinomialClassifierSegment()
        multi_classifier.multinomail_classifier(X_train=np_x_train, y_train=Y_train, doc_vectorizer=doc_vectorizer, model_no=model_no,model_path=model_path)
        multinomial_prediction_df = prediction_df.copy(deep=True)
        multinomial_prediction_df=multi_classifier.predict_from_model(df=multinomial_prediction_df,model_no=model_no,model_path=model_path,X_test=np_x_test, y_test=Y_test)
        CsvOperation().classifier_model_df_to_csv(df=multinomial_prediction_df, model_no=model_no, model_path=model_path,
                         model_name="multinomial_navie_bayse")

        decision_classifier = DecisionTreeSegment()
        decision_classifier.decision_tree_classifier(X_train=np_x_train, y_train=Y_train, doc_vectorizer=doc_vectorizer, model_no=model_no,model_path=model_path)
        decision_prediction_df = prediction_df.copy(deep=True)
        decision_prediction_df = decision_classifier.predict_from_model(df=decision_prediction_df,model_no=model_no,model_path=model_path,X_test=np_x_test, y_test=Y_test,X_train=np_x_train,y_train=Y_train)
        # prediction_df=prediction_df.join(decision_prediction_df)
        CsvOperation().classifier_model_df_to_csv(df=decision_prediction_df,model_no=model_no,model_path=model_path,model_name="decision_tree")

        knn_classifier = KnnClassifierSegment()
        knn_classifier.kneighbors_classifier(X_train=np_x_train, y_train=Y_train, doc_vectorizer=doc_vectorizer,
                                                     model_no=model_no, model_path=model_path)
        knn_prediction_df = prediction_df.copy(deep=True)
        knn_prediction_df= knn_classifier.predict_from_model(df=knn_prediction_df,model_no=model_no,model_path=model_path,X_test=np_x_test, y_test=Y_test)
        CsvOperation().classifier_model_df_to_csv(df=knn_prediction_df, model_no=model_no, model_path=model_path,
                         model_name="knn")

        # predict_accuracy = bernoulli_classifier(training_data=training_data,test_data=test_data,cross_validation_data=cross_test_data,doc_vectorizer=doc_vectorizer,technology_column=technology_column,model_no=model_no,model_path=model_path)
        # accuracy_df = accuracy_df.append(pd.DataFrame([['navie_bayes_bernoulli','tfidf',model_name,predict_accuracy]],columns=columns),ignore_index=True)

def ensembling_voting(X_train=None, X_test=None, Y_train=None, Y_test=None):
    """

    :param X_train:
    :param X_test:
    :param Y_train:
    :param Y_test:
    :return:
    """
    clf1 = DecisionTreeClassifier(max_depth=5)
    clf2 = RandomForestClassifier()
    clf3 = RandomForestClassifier(criterion='entropy', max_depth=4)
    # clf4 = MultinomialNB()
    clf5 = KNeighborsClassifier(weights="distance", algorithm='kd_tree')
    clf6 = KNeighborsClassifier(weights="uniform", leaf_size=20)
    clf7 = svm.SVC()
    clf_list = [('decision_tree', clf1), ('random_foresrt', clf2), ('random_forest_information_gain', clf3),
                ('knn', clf5), ('knn2', clf6),('svm',clf7)]
    weight = [1,2,1,1,2,2]
    vectorizer = TfidfVectorizer(max_df=0.4, min_df=0.2,
                                 stop_words='english')
    # vectorizer = CountVectorizer(ngram_range=(1,2),stop_words='english',max_df=0.7,min_df=0.2)
    # vectorizer = HashingVectorizer()
    np_x_train = X_train['processed_text']
    np_x_test = X_test['processed_text']
    evc =EnsemblingMechanism().ensembling_voting_classifier(X_train=np_x_train, y_train=Y_train, X_test=np_x_test,
                                                       y_test=Y_test, estimator_classifier=clf_list,
                                                       vectorizer=vectorizer,weight=weight)
    EnsemblingMechanism().ensembling_voting_classifier_prediction(evc=evc,X_test=np_x_test,y_test=Y_test)


def ensembling_weight_classifiers(X_train=None, X_test=None, Y_train=None, Y_test=None):
    """

    :param X_train:
    :param X_test:
    :param Y_train:
    :param Y_test:
    :return:
    """

    vectorizer = TfidfVectorizer(max_df=0.4, min_df=0.2,
                                 stop_words='english')
    # vectorizer = CountVectorizer(ngram_range=(1,2),stop_words='english',max_df=0.7,min_df=0.2)
    # vectorizer = HashingVectorizer()
    np_x_train = X_train['processed_text']
    np_x_test = X_test['processed_text']
    EnsemblingMechanism().ensembling_weight_classification_prediction(X_train=np_x_train, Y_train=Y_train,
                                                                      X_test=np_x_test,Y_test=Y_test,vectorizer=vectorizer)

if __name__ == '__main__':
    technology_column = "technology_segment_2"
    model_path = r"/Users/nitesh/OneDrive/Work/GE_Python_Workspace/ClassifierApproaches/ClassifierModels/Node3/"
    df = pd.read_csv(r"/Users/nitesh/OneDrive/Work/GE_Python_Workspace/ClassifierApproaches/Excel Documents/"
                     r"Fintech/Fintech_Digital_Banking_Training_Data.csv")
    df, X, Y= DataPreprocessing().get_complete_text(df=df,technology_column=technology_column)
    # CustomSplits().check_stratify_split(X=X,y=Y)
    labels = Y[technology_column]
    X_train, X_test, Y_train, Y_test = CustomSplits().random_X_and_Y(X=X, Y=labels,random_number=90)
    GEPlots().count_of_each_categories(y_train=Y_train,y_test=Y_test,technology_segment=technology_column)
    # ensembling_voting(X_train, X_test, Y_train, Y_test)
    # accuracy_df = training_and_modeling(df=df, X=X, Y=Y, technology_column=technology_column, model_path=model_path)
    # # accuracy_df.to_csv('/Users/nitesh/Documents/GE_Work_Documents/GE_DATA_CSV/Accuracy_model_perc/approach3/model_accuracy.csv')
    # print(accuracy_df)


