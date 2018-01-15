#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 16:07:36 2017

@author: nitesh
"""
import pickle
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from string import punctuation
import pandas as pd
from Preprocessing.DataMassaging import DataPreprocessing
import numpy as np
import random
import math
import os



_stopwords = set(stopwords.words('english') + list(punctuation))
_stemmer = SnowballStemmer('english')




class BestClassifier():
    """

    """

    def csv_to_df(self,path=None):
        """
        Converts from the csv to data frame
        :param path: Expects the path/location of the csv file
        :return data frame
        """
        # reads the csv file and puts it to the dataframe
        df = pd.read_csv(path)
        return df

    def nltk_preprocessing(self,content):
        """
        simple preprocessing step is taken care.
        1> converts the words to lowercase.
        2> removes the stopwords
        3> stems all the words

        :param content: Expects the raw doc
        :return preprocessed text
        """

        text = ""
        tokens = word_tokenize(content)
        for token in tokens:
            token = token.lower()
            if token not in _stopwords:
                stemmed_token = _stemmer.stem(token)
                if text == "":
                    text = stemmed_token
                else:
                    text = text + ' ' + stemmed_token
        return text

    def get_best_classifier_model(self,pickle_path=None):
        """

        :param pickle_path:
        :return:
        """
        pickle_in = open(pickle_path, "rb")
        model = pickle.load(pickle_in)
        pickle_in.close()
        return model

    # def random_X(self,X=None):
    #     """
    #
    #     :param X:
    #     :param Y:
    #     :return:
    #     """
    #     random_number = random.randint(1, 101)
    #     print("The random number is ", random_number)
    #     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_number, stratify=Y)
    #     return X_train, X_test, Y_train, Y_test

    def get_complete_text(self,df=None):
        """
        Takes the specialties, short_desc, about us and concatinates and provides a complete contacatated result in the df
        :param df: Need the df which has the above mentioned columns
        :return dataframe
        """
        df['complete_text'] = df[['short_description', 'about_us']].apply(
            lambda x: x.str.cat(sep=' '), axis=1)
        df['processed_text'] = df['complete_text'].apply(DataPreprocessing().nltk_preprocessing)
    #    X = df.iloc['processed_text']
    #    Y = df.iloc[technology_column]
        X = pd.DataFrame()
        X['processed_text'] = df['processed_text']
        X['domain_name'] = df['domain_name']
        return df, X

    def predict_from_model(self,df=None,data_name= None, model=None,X_test=None):
        """

        :param df:
        :param model_no:
        :param model_path:
        :param X_test:
        :param y_test:
        :return:
        """
        predicted_test_result = model.predict(X_test)

        df[data_name] = predicted_test_result
        return df

    def new_test_dataset(self,data1_model=None,data2_model=None,X_test=None):
        """

        :param data1_model:
        :param data2_model:
        :param X_test:
        :return:
        """
        np_x_test = X_test['processed_text']

        prediction_df = pd.DataFrame()
        prediction_df['domain_name'] = X_test['domain_name']
        prediction_df['Test_document'] = X_test['processed_text']

        clone_df1 = prediction_df.copy(deep=True)
        clone_df1 = self.predict_from_model(df=clone_df1, data_name="predicted_technology",
                                                                        model=data1_model, X_test=np_x_test)
        # df = self.predict_from_model(df=clone_df1, data_name="data_segment2",
        #                                                                 model=data2_model, X_test=np_x_test)
        save_path = r"/Users/nitesh/OneDrive/Work/GE_Python_Workspace/ClassifierApproaches/Excel Documents/Fintech/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        csv_export_path = save_path + "fintech_digital_banking_prediction"+".csv"
        clone_df1.to_csv(csv_export_path)

        # df.to_csv(csv_export_path)

    def new_test_dataset_for_multi_models(self,dict_of_models=None,X_test=None):
        """

        :param model_list:
        :param X_test:
        :return:
        """
        np_x_test = X_test['processed_text']

        prediction_df = pd.DataFrame()
        prediction_df['domain_name'] = X_test['domain_name']
        prediction_df['Test_document'] = X_test['processed_text']

        clone_df1 = prediction_df.copy(deep=True)
        for key, value in dict_of_models.items():
            clone_df1 = self.predict_from_model(df=clone_df1, data_name=key,
                                                model=value, X_test=np_x_test)
            print("prediction for model name ",key," is done")
        save_path = r"/Users/nitesh/OneDrive/Work/GE_Python_Workspace/ClassifierApproaches/Excel Documents/Fintech/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        csv_export_path = save_path + "fintech_digital_banking_prediction"+".csv"
        clone_df1.to_csv(csv_export_path)

    def initiate(self):
        """

        :return:
        """
        model1_path = r"/Users/nitesh/OneDrive/Work/GE_Python_Workspace/ClassifierApproaches/" \
                          r"Model_Accuracy_csv/Fintech/Digital_banking/decision/decision_tree_model114.pickle"
        # model2_path = r"/Users/nitesh/OneDrive/Work/GE_Python_Workspace/ClassifierApproaches/ClassifierModels/ML_SEGMENT_DATA2_AP4_T2_1RUN_MODELS/Model6/KNeighborClassifierTfidf6.pickle"
        data1_model = self.get_best_classifier_model(pickle_path=model1_path)
        # data2_model = self.get_best_classifier_model(pickle_path=model2_path)
        df = self.csv_to_df(
            path=r"/Users/nitesh/OneDrive/Work/GE_Python_Workspace/ClassifierApproaches/Excel Documents/"
                     r"Fintech/Fintech_Digital_Banking_Test_Data.csv")
        df, X = self.get_complete_text(df=df)
        X_test = X[['domain_name', 'processed_text']]
        self.new_test_dataset(data1_model=data1_model,X_test=X_test)

    def initiate_all_best_models(self):


        """

        :return:
        """
        models_path = r"/Users/nitesh/OneDrive/Work/GE_Python_Workspace/ClassifierApproaches/" \
                      r"Model_Accuracy_csv/Fintech/Digital_banking/decision/"
        df = self.csv_to_df(
            path=r"/Users/nitesh/OneDrive/Work/GE_Python_Workspace/ClassifierApproaches/Excel Documents/"
                 r"Fintech/Fintech_Digital_Banking_Test_Data.csv")
        dict_model = dict()
        for file in os.listdir(models_path):
            if file.endswith(".pickle"):
                print(os.path.join(models_path, file))
                model_pickle_path = os.path.join(models_path, file)
                model = self.get_best_classifier_model(pickle_path=model_pickle_path)
                dict_model[file] = model


        df, X = self.get_complete_text(df=df)
        X_test = X[['domain_name', 'processed_text']]
        self.new_test_dataset_for_multi_models(dict_of_models=dict_model, X_test=X_test)

if __name__ == '__main__':
    BestClassifier().initiate_all_best_models()