#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 16:00:01 2018

@author: nitesh
"""

import os
import pandas as pd
import math
import logging
from LoggingDetails.LogFileHandler import logs
from LoggingDetails.LogPathConstant import LogFilePathContants,LogFiles
from CSVUtilities.CSVPathConstants import CSV_NAME,CSVFilePathContants
log_file = os.path.join(LogFilePathContants._BASE_LOG_FILE,
                        LogFiles._CSV_LOG_FILE)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logs().fileHandler(logging,log_file_path=log_file)
logger.addHandler(file_handler)
class CSVToDictionaryMapping():

    def __init__(self):
        self._csv_name = CSV_NAME()
        self._file_path = CSVFilePathContants()

    @property
    def file_path(self):
        return self._file_path

    @file_path.setter
    def file_path(self,value):
        self._file_path = value

    @property
    def csv_value(self):
        return self._csv_name

    @csv_value.setter
    def csv_value(self, value):
        self.csv_value = value

    def dict_to_df(self,dictionary=dict()):
        """

        :param dictionary:
        :return:
        """
        df = pd.DataFrame.from_dict(dictionary,orient="index")
        return df

    def df_to_dict(self,df=pd.DataFrame()):
        """

        :param df:
        :return:
        """
        d = df.to_dict()
        tech_dict = {}
        for key, value in d.items():
            tech_name_list = list()
            for i, val in value.items():
                if isinstance(val, float):
                    logger.warning("The csv contains %s nan under key %s index %s",str(math.isnan(val)) ,str(key),str(i))
                elif (str(val)!=None):
                    tech_name_list.append(val)

            tech_dict[key]=tech_name_list
        return tech_dict

    def get_filter_technology_to_df(self):
        """

        :return:
        """
        csv_path = os.path.join(self.file_path._BASE_DOC_PATH,
                                self.csv_value._CLUSTER_COLUMN_CSV)
        logger.info('The csv path is {}'.format(csv_path))
        try:
            df = pd.read_csv(csv_path)
            return df
        except (SystemExit, KeyboardInterrupt):
            raise
        except FileNotFoundError as error:
            logger.error("check the file existence {}".format(str(error)))
        except Exception as error:
            logger.error("Failed to open file, check file and the name of the file {}".format(str(error)),exc_info=True)

    def dict_to_df(self,dictionary=dict()):
        """

        :param dictionary:
        :return:
        """
        df = pd.DataFrame.from_dict(dictionary,orient="index")
        return df

