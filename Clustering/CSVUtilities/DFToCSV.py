#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 16:55:19 2018

@author: nitesh
"""

import os
import pandas as pd
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


class DataframeToCSV():
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

    def load_df_to_csv(self, df=pd.DataFrame(),cluster_csv=None,algo=None):
        """

        :param df:
        :return:
        """
        csv_path = None
        if(cluster_csv=="cluster_tag"):
            if(algo=="kmeans"):
                csv_path = os.path.join(self.file_path._BASE_DOC_PATH,
                                        self.csv_value._CLUSTER_TAG_CSV) #This the csv path of GE
                logger.info('The csv path is {}'.format(csv_path))
            elif(algo=="hierarchy"):
                csv_path = os.path.join(self.file_path._BASE_DOC_PATH,
                                        self.csv_value._HC_CLUSTER_TAG_CSV)
                logger.info('The csv path is {}'.format(csv_path))

        elif(cluster_csv=="cluster_filename"):
            if (algo == "kmeans"):
                csv_path = os.path.join(self.file_path._BASE_DOC_PATH,
                                        self.csv_value._CLUSTER_FILENAME_CSV)  # This the csv path of GE
                logger.info('The csv path is {}'.format(csv_path))

            elif (algo == "hierarchy"):
                csv_path = os.path.join(self.file_path._BASE_DOC_PATH,
                                        self.csv_value._HC_CLUSTER_FILENAME_CSV)
                logger.info('The csv path is {}'.format(csv_path))

            elif (algo == "affinity_propagation"):
                csv_path = os.path.join(self.file_path._BASE_DOC_PATH,
                                        self.csv_value._AFFINITY_CLUSTER_FILENAME_CSV)
                logger.info('The csv path is {}'.format(csv_path))

        elif(cluster_csv=="cluster_distance"):
            if(algo == "affinity_propagation"):
                csv_path = os.path.join(self.file_path._BASE_DOC_PATH,
                                        self.csv_value._AFFINITY_CLUSTER_DISTANCE_CSV)
                logger.info('The csv path is {}'.format(csv_path))


        elif (cluster_csv == "cluster_column"):
            csv_path = os.path.join(self.file_path._BASE_DOC_PATH,
                                    self.csv_value._CLUSTER_COLUMN_CSV)  # This the csv path of GE
            logger.info('The csv path is {}'.format(csv_path))


        try:
            logger.info("Saving to the CSV PATH : {}".format(str(csv_path)))
            df.to_csv(csv_path)
            logger.info("data frame to csv conversion done")
        except (SystemExit, KeyboardInterrupt):
            raise
        except FileNotFoundError as error:
            logger.error("check the file existence {}".format(str(error)))
        except Exception as error:
            logger.error("Failed to open file, check file and the name of the file {}".format(str(error)),exc_info=True)

