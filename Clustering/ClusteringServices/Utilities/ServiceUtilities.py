#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 14:37:19 2018

@author: nitesh
"""

import logging
from LoggingDetails.LogFileHandler import logs
from LoggingDetails.LogPathConstant import LogFilePathContants,LogFiles
from CSVUtilities.CSVAndDFToDict import CSVToDictionaryMapping
import os


log_file = os.path.join(LogFilePathContants._BASE_LOG_FILE,
                        LogFiles._CLUSTERING_LOG_FILE)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logs().fileHandler(logging,log_file_path=log_file)
logger.addHandler(file_handler)



class ExtraTools():
    def filter_input(self):
        filter_dict = dict()
        while True:
            print("filtered dict... press 1 to continue 2 to break")
            temp = input()
            if temp == "1":

                print("Enter the field name")
                key = str(input())
                print(key)
                print("Enter the value")
                value = str(input())
                print(value)
                val = list()
                if key in filter_dict:
                    val = filter_dict.get(key)
                    val.append(value)
                    filter_dict[key] = val
                else:
                    val.append(value)
                    filter_dict[key] = val

            else:
                break;
        return filter_dict

    def get_filtered_dict(self):
        """

        :param k_means:
        :param dict:
        :return:
        """
        csv_dict = CSVToDictionaryMapping()
        df_column = csv_dict.get_filter_technology_to_df()
        filtered_tech_dict = csv_dict.df_to_dict(df=df_column)

        logger.info("the df format of the filter csv {}".format(str(df_column)))
        logger.info("converted df to dict {}".format(str(filtered_tech_dict)))

        return filtered_tech_dict

    def get_number_of_categories(self,unique_values_dict_categories=None):
        """

        :param unique_values_dict_categories:
        :return:
        """
        count_of_categories = 0
        if(unique_values_dict_categories!=None):
            for key, value in unique_values_dict_categories.items():
                count_of_categories += len(value)
            logger.info("the total count of the categories is: {}".format(str(count_of_categories)))
            return count_of_categories
        else:
            logger.info('No categories are found from the unique values of dict')