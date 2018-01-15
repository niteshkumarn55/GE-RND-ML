#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 15:12:36 2017

@author: nitesh
"""
import pandas as pd

class CsvOperation():


    def csv_to_df(self, path=None):
        """
        Converts from the csv to data frame
        :param path: Expects the path/location of the csv file
        :return data frame
        """
        # reads the csv file and puts it to the dataframe
        df = pd.read_csv(path)
        return df

    def classifier_model_df_to_csv(self, df=None, model_no=None, model_path=None, model_name=None):
        """

        :param df:
        :param model_no:
        :param model_path:
        :param model_name:
        :return:
        """
        path = model_path + "Model" + str(model_no)
        csv_export_path = path + "/accuracy_csv_model_" + model_name + "_" + str(model_no) + ".csv"
        df.to_csv(csv_export_path)




