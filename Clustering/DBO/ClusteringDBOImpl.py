#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 14:54:19 2018

@author: nitesh
"""
import os
import logging
import pandas as pd
import mysql.connector
import sqlalchemy
from DBO.DataBaseServices import DBConnection
from LoggingDetails.LogFileHandler import logs
from LoggingDetails.LogPathConstant import LogFilePathContants, LogFiles
from sqlalchemy import update


log_file = os.path.join(LogFilePathContants._BASE_LOG_FILE,
                        LogFiles._DB_LOG_FILE)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logs().fileHandler(logging, log_file_path=log_file)
logger.addHandler(file_handler)

class ClusterCurd():

    def insert_job_id_and_cluster_name(self,df=pd.DataFrame):
        """

        :param df:
        :return:
        """
        if(len(df)>0):
            db = DBConnection()
            conn = db.get_connection()
            df.to_sql(name='analyst_console_job_cluster_mapping', con=conn, if_exists='append', index=False) #Index = false for not letting the index to save as a column in the DB
            logger.info("Append to the tables job_cluster_mapping, map of job and cluster name is done successfully...")
            db.close_connection(con=conn)
        else:
            logger.error("It must be empty df.... hence DB insertion was aborted")

    def get_job_tbl_by_jobid(self, job_id=None):
        """

        :param job_id:
        :return:
        """
        if (job_id != None):
            db = DBConnection()
            conn = db.get_connection()
            result = conn.execute("select * from analyst_console_job_table where job_id = " + job_id)
            db.close_connection(con=conn)
            logger.info("the results from the table analyst_console_job_table is retrived for job_id {}".format(str(job_id)))
            return result
        else:
            logger.error(
                "Something wrong in fetching data from job_cluster_mapper from the job_id {}".format(str(job_id)))

    def get_cluster_id_by_jc_id(self,job_id=None):
        """

        :param job_id:
        :return:
        """
        if(job_id!=None):
            db = DBConnection()
            conn = db.get_connection()
            result = conn.execute("select * from analyst_console_job_cluster_mapping where  job_id = " +job_id)
            db.close_connection(con=conn)
            logger.info(
                "the results from the table analyst_console_job_cluster_mapping is retrived for job_id {}".format(str(job_id)))
            return result
        else:
            logger.error("Something wrong in fetching data from job_cluster_mapper from the job_id {}".format(str(job_id)))



    def insert_jc_id_and_filename(self,df=pd.DataFrame):
        """

        :param df:
        :return:
        """
        if(len(df)>0):
            db = DBConnection()
            conn = db.get_connection()
            df.to_sql(name='analyst_console_cluster_company_mapping', con=conn, if_exists='append', index=False)
            logger.info("Append to the tables cluster_and_company_mapping is done successfully")
            db.close_connection(con=conn)
        else:
            logger.error("It must be empty df.... hence DB insertion was aborted")

    def update_status_in_job_table(self,status=None, job_id=None):
        """

        :param status:
        :param job_id:
        :return:
        """
        if(status!=None and job_id!=None):
            logger.info("Update of job table is initiated....")
            engine = sqlalchemy.create_engine("mysql://Growthenabler:G30WthEn@813r@88.198.208.55/dev_growth_enabler_ui")
            md = sqlalchemy.MetaData(engine)
            table = sqlalchemy.Table('analyst_console_job_table', md, autoload=True)
            stmt = update(table).where(table.c.job_id == job_id). \
                values(status='done')
            engine.execute(stmt)
            # stmt = update('analyst_console_job_table', 'analyst_console_job_table'.job_id==job_id)
            # conn.execute(stmt,status="Done")


