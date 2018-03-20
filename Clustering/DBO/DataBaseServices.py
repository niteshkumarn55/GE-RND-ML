#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Fri 09 18:30:14 2018

@author: nitesh
"""

import os
import logging
import pandas as pd
import mysql.connector
from sqlalchemy import create_engine
from LoggingDetails.LogFileHandler import logs
from LoggingDetails.LogPathConstant import LogFilePathContants, LogFiles


log_file = os.path.join(LogFilePathContants._BASE_LOG_FILE,
                        LogFiles._DB_LOG_FILE)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logs().fileHandler(logging, log_file_path=log_file)
logger.addHandler(file_handler)

class DBConnection():


    def get_connection(self):
        """

        :return: db: Gets the connection for the db dev_growth_enabler_ui
        """
        logger.info("Establishing the connection with the DB")
        # engine = sqlalchemy.create_engine('mysql://user:password@server')  # connect to server
        engine = create_engine("mysql://Growthenabler:G30WthEn@813r@88.198.208.55/dev_growth_enabler_ui")

        con = engine.connect()
        logger.info("Connection established with the DB, succusfully ")
        return con

    def close_connection(self, con):
        logger.info("Establishing the operation to close the connection with DB")
        con.close()
        logger.info("Connection closed with the DB")

class DFToSQl():

    def save_df_to_sql(self,df=pd.DataFrame()):
        """

        :return:
        """

        engine = create_engine("mysql://root:root@12345@localhost/GE_CLUSTERS")
        con = engine.connect()
        data = {'A':['1','3','4','8'],'B':['2','4','7','9']}
        df = pd.DataFrame(data=data)
        df['id'] = df.index
        df.to_sql(name='tbl_cluster', con=con, if_exists='append')

if __name__ == '__main__':
    db = DFToSQl()
    db.save_df_to_sql()
