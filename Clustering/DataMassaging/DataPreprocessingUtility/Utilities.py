#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 16:17:19 2018

@author: nitesh
"""
import logging
# DEBUG: Detailed information, typically of interest only when diagnosing problems.
# INFO:	Confirmation that things are working as expected.
# WARNING: An indication that something unexpected happened, or indicative of some problem in the near future
# (e.g. ‘disk space low’). The software is still working as expected.
# ERROR: Due to a more serious problem, the software has not been able to perform some function.
# CRITICAL:	A serious error, indicating that the program itself may be unable to continue running.
# Display progress logs on stdout
import os
import pandas as pd
from LoggingDetails.LogFileHandler import logs
import numpy as np
from LoggingDetails.LogPathConstant import LogFilePathContants,LogFiles
log_file = os.path.join(LogFilePathContants._BASE_LOG_FILE,
                        LogFiles._DATA_MASSAGE_LOG_FILE)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logs().fileHandler(logging,log_file_path=log_file)
logger.addHandler(file_handler)

class DataUtilities():

    def __init__(self,df=pd.DataFrame()):
        self._df = df

    def show_hori_tech_from_df(self):
        """
        This function provides the df which is required for the classification/clustering problem.
        It removes all the NaN from the independent columns.
        The returned df precisely contains data for the classification/clustering
        This also find the independent columns/fields inside the df
        :param df: take the entire dataframe, The date from the csv to df should contain indenpendent column names as
                    'horizontal','technology_segment_1','technology_segment_2' etc...
        :return: required_df: required_df contains only the df which is taken for classification/clustering
        :return: Indenpendent_fields: This returns the column names which are considered as the indenpendent values
        """
        independent_fields = list()
        try:
            if(len(self._df)<=0):
                logger.error("Data frame is not a valid data frame")
                raise
            else:

                # Filtering on the data needed for the classification/clustering
                try:
                    required_df = self._df.filter(['ID', 'company_name', 'domain_name','short_description','about_us'], axis=1)
                except Exception as error:
                    logger.error("Filtering the data frame caused problem ",str(error),exc_info=True)

                column_names = list(self._df.columns.values) #This gets all the column names from the df

                for value in column_names:
                    if(value.startswith('horizontal') or value.startswith('technology')): #dynamically selecting the independent values
                        independent_fields.append(value)

                        #Gets only the required columns to the dataframe
                        required_df[value] = self._df[value]
                        required_df = required_df[required_df[value].notnull()] #Just picking rows which has no NaN
                logger.info("The filtered df is {}".format(str(required_df.head(1))))
                logger.info("The independent values are {}".format(str(independent_fields)))
                return required_df, independent_fields
        except Exception as error:
            logger.error(str(error),exc_info=True)


class DFAnalyse():

    def get_unique_fields_based_on_tech_and_horizontal(self,df=pd.DataFrame(),independent_fields=list()):
        """
        Gets the unique values from all the indenpendent columns. You need to provide who
        all are the independent fields/columns

        :param df: give the df by which the unique value for the independent values should be extracted
        :param independent_fields: takes the independent column names Ex: ['horizontal','technology_segment_1',..] etc..
        :return: unique_values_dict: returns a dict of independent column names and it unique set of values
                example, {'horizontal': ['Fintech'], 'technology_segment_1': ['Digital Banking'],
                'technology_segment_2': ['Open Banking API', 'Ommi Channel Banking',
                'Customer Relationship Management', 'Analytics']}
        """
        try:
            if(len(df)<=0 or len(independent_fields)<=0):
                logger.error("Data frame is not a valid data frame or the independent fields is 0")
                raise
            else:
                unique_values_dict = dict()
                for val in independent_fields:
                    #Gets the unique values from all the independent columns
                    list_unique_values = pd.unique(df[[val]].values.ravel('K')).tolist()
                    unique_values_dict[val] = list_unique_values
                    logger.info("The unique value of dict for key '{}' and the list of values is '{}'".
                                format(str(val),str(list_unique_values)))
                return unique_values_dict
        except Exception as error:
            logger.error(str(error),exc_info=True)

    def filter_df_by_category(self,df=pd.DataFrame(),dict_category_and_values=dict()):
        """
        This filters the df by the categories you pick
        :param df: takes the df entire values and columns
        :param dict_category_and_values: takes the dict of column name/independent fields
                                        and the value for which it needs to be filtered
                                        example, temp_dict = {'horizontal': ['Fintech'],
                                        'technology_segment_2': ['Open Banking API',
                                        'Customer Relationship Management']}
        :return: cate_filtered_df: returns the df which is filtered based on the categories and been given in the
                                    dict_category_and_values
        """
        try:
            if (len(df) <= 0 or len(dict_category_and_values) <= 0):
                logger.error("Data frame is not a valid data frame or the dict_category_and_values fields lenght is 0")
                raise
            else:
                cate_filtered_df = df.copy(deep=True)
                for key, unique_value_list in dict_category_and_values.items():
                    if (key == "all"):
                        pass
                        logger.info("Going to take all the fields from the csv to df")
                        break
                    else:
                        #Filters the df based on the independent field of its categorical value
                        cate_filtered_df = cate_filtered_df.loc[cate_filtered_df[key].isin(unique_value_list)]
                        logger.info("The df is forming based on the column/independent field {} and the categories "
                                    "selected is {}".format(str(key),str(unique_value_list)))
                return cate_filtered_df
        except Exception as error:
            logger.error(str(error), exc_info=True)
#
# if __name__ == '__main__':
#
#     filter_df, independent_fields = data_utilities.show_hori_tech_from_df(df=df)
#     dfa = DFAnalyse()
#     dfa.get_unique_fields_based_on_tech_and_horizontal(df=filter_df,independent_fields=independent_fields)
#
#     # temp_dict = {'horizontal': ['Fintech'], 'technology_segment_1': ['Digital Banking'], 'technology_segment_2': ['Open Banking API', 'Customer Relationship Management']}
#     temp_dict = {'horizontal': ['Fintech'],'technology_segment_2': ['Open Banking API', 'Customer Relationship Management']}
#
#     dfa.filter_df_by_category(df=filter_df,dict_category_and_values=temp_dict)