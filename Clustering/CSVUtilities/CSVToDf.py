#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 16:00:01 2018

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


class CsvToDataFrame():

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


    def get_df_from_csv(self, job_id=None):
        """
        This method provides the df from the csv present in the default path.
        This also removes the rows where the domain_name are NaN
        :return: df: DF which is converted from the csv format
        """
        if(job_id!=None):
            csv_name = job_id+".csv"
            csv_path = os.path.join(self.file_path._BASE_DOC_PATH,csv_name)
        else:
            csv_path = os.path.join(self.file_path._BASE_DOC_PATH,
                                    self.csv_value._GE_SEGMENT_CSV) #This the csv path of GE

        logger.info('The csv path is {}'.format(csv_path))
        try:
            df = pd.read_csv(csv_path)
            logger.info("Dataframe is create %s is the same of record",df.head(1))
            df_n = df[df['domain_name'].isnull()]
            if df_n.size > 0:
                logger.warning('The ID of Records where Domain name has NULL, {} '.format(str(df_n['ID'].values)))
            logger.info('Storing Non Null value in the dataframe')
            cols = ['domain_name','short_description','about_us']
            for col in cols:
                df = df[df[col].notnull()] #Just picking rows which has no NaN in domain_name
            return df
        except (SystemExit, KeyboardInterrupt):
            raise Exception
        except FileNotFoundError as error:
            logger.error("check the file existence {}".format(str(error)))
        except Exception as error:
            logger.error("Failed to open file, check file and the name of the file {}".format(str(error)),exc_info=True)
