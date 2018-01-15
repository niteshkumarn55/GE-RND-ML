#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 12:20:15 2017

@author: nitesh
"""
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table


class CustomSplits():
    """

    """

    def check_stratify_split(self,X=None, y=None, Y_train=None, Y_test=None,check=None):

        if check == None:
            X_train, X_test, Y_train, Y_test = self.random_X_and_Y(X=X, Y=y)
            print("The count of the Y train is", Y_train.size)
            print(type(Y_train))
            print("The lenght of Y train",len(Y_train))
            print("The lenght of Y test", len(Y_test))
            unique_values = y.ml_segment_data2_ap4_t2.unique()
            for i in unique_values:
                len_of_each_segements_in_y_train = len(Y_train[Y_train['ml_segment_data2_ap4_t2'] == i])
                len_of_all_segements_in_total_y = len(y[y['ml_segment_data2_ap4_t2'] == i])
                print("The lenght of ", i, "in the panda dataset is ", len_of_each_segements_in_y_train)
                print("The lenght of ", i, "in the panda dataset is ", len_of_all_segements_in_total_y)
                percentage_of_segment_split_training = (
                                                           len_of_each_segements_in_y_train / len_of_all_segements_in_total_y) * 100
                print("the percentage of train split is ", percentage_of_segment_split_training)


        elif check == 'train_data':
            print("The count of the Y train is", Y_train.size)
            print(type(Y_train))
            print(len(Y_train))
            unique_values = Y_train.ml_segment_data2_ap4_t2.unique()
            for i in unique_values:
                len_of_each_segements_in_y_train = len(Y_train[Y_train['ml_segment_data2_ap4_t2'] == i])
                len_of_all_segements_in_total_y = len(y[y['ml_segment_data2_ap4_t2'] == i])
                print("The lenght of ", i, "in the panda dataset is ", len_of_each_segements_in_y_train)
                print("The lenght of ", i, "in the panda dataset is ", len_of_all_segements_in_total_y)
                percentage_of_segment_split_training = (
                                                       len_of_each_segements_in_y_train / len_of_all_segements_in_total_y) * 100
                print("the percentage of train split is ", percentage_of_segment_split_training)

        elif check == 'test_data':
            print("The count of the Y train is", Y_test.size)
            print(type(Y_train))
            print(len(Y_train))
            unique_values = Y_test.ml_segment_data2_ap4_t2.unique()
            for i in unique_values:
                len_of_each_segements_in_y_test = len(Y_test[Y_test['ml_segment_data2_ap4_t2'] == i])
                len_of_all_segements_in_total_y = len(y[y['ml_segment_data2_ap4_t2'] == i])
                print("The lenght of ", i, "in the panda dataset is ", len_of_each_segements_in_y_test)
                print("The lenght of ", i, "in the panda dataset is ", len_of_all_segements_in_total_y)
                percentage_of_segment_split_test = (
                                                           len_of_each_segements_in_y_test / len_of_all_segements_in_total_y) * 100
                print("the percentage of train split is ", percentage_of_segment_split_test)


    def random_X_and_Y(self, X=None, Y=None,random_number=None):
        """

        :param X:
        :param Y:
        :return:
        """
        if random_number == None:
            random_number = random.randint(1, 200)

        print("The random number is ", random_number)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_number, stratify=Y)
        return X_train, X_test, Y_train, Y_test

    def shuffle_by_category(self, df=None, technology_column=None):
        """
        shuffles the dataset into training and test set based on certain percentage in every categories.
        :param: df : dataframe
        :param: technology_column : independent column name
        :return:
        """
        # Number of training and testing records distribution percentage
        train_percentage = random.randint(79, 81)
        test_percentage = 100 - train_percentage

        # Creating pandas for the training and testing set
        training_data = pd.DataFrame()
        test_data = pd.DataFrame()

        # category wise equally collecting data for training and testing dataset. This make sures that each category has sufficent amount of data for training and test dataset
        for cate_segment in list(df[technology_column].unique()):
            temp_df = pd.DataFrame()
            temp_df = df.loc[df[technology_column] == cate_segment]

            data_len = len(temp_df)
            train_records = data_len * (train_percentage / 100)
            train_records = math.ceil(train_records)
            test_records = data_len - train_records

            # Shuffling the data w.r.t category
            temp_df = temp_df.iloc[np.random.permutation(len(temp_df))]
            temp_train = temp_df[['domain_name', 'processed_text', technology_column]].head(train_records)
            temp_test = temp_df[['domain_name', 'processed_text', technology_column]].tail(test_records)

            training_data = pd.concat(
                [training_data, temp_train])  # Concatenates the dataframe below the other dataframe
            test_data = test_data.append(temp_test)  # Append doesn't work inplace you need to store the output

            X_train = training_data['processed_text']
            y_train = training_data[technology_column]
            X_test = test_data['processed_text']
            y_test = test_data[technology_column]

        return X_train, X_test, y_train, y_test




