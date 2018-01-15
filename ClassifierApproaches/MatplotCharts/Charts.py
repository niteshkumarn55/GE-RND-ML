#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 4 11:30:36 2018

@author: nitesh
"""
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table
class GEPlots():

    def count_of_each_categories(self,y_train=None,y_test=None,technology_segment=None):
        """
        Shows the plot of the training and test split data
        :param y_train: Takes the Y_train data. Used to train the model
        :param y_test: Takes the Y_test data. This is the data which will be further used to find the accuracy of the model.
        :return: saves the plot in a particular path
        """
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()

        # Converting into proper pd dataframe
        df_train[technology_segment] = y_train
        df_test[technology_segment] = y_test

        #Getting the value count of each category/technology segment for training data
        df = pd.DataFrame()
        df['freq'] = df_train[technology_segment].value_counts()

        # ax = plt.subplot(111, frame_on=False)  # no visible frame
        # ax.xaxis.set_visible(False)  # hide the x axis
        # ax.yaxis.set_visible(False)  # hide the y axis

        #Plotting the a graph of training data split
        fig, ax = plt.subplots(1, 1)
        table(ax, df,
               loc = 'upper right', colWidths = [0.2, 0.2, 0.2])
        df.plot(ax=ax)
        # table(ax, df,cellLoc = 'center', rowLoc = 'center',
        #   loc='down')  # where df is your data frame
        # plt.show()
        plt.savefig(r"/Users/nitesh/OneDrive/Work/GE_Python_Workspace/ClassifierApproaches/Excel Documents/"
                     r"Fintech/train_data_split.png")

        # Getting the value count of each category/technology segment for test data
        df = pd.DataFrame()
        df['freq'] = df_test[technology_segment].value_counts()

        # Plotting the a graph of test data split
        fig, ax = plt.subplots(1, 1)
        table(ax, df,
              loc='upper right', colWidths=[0.2, 0.2, 0.2])
        df.plot(ax=ax)

        plt.savefig(r"/Users/nitesh/OneDrive/Work/GE_Python_Workspace/ClassifierApproaches/Excel Documents/"
                    r"Fintech/test_data_split.png")

