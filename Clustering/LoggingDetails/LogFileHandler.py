#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 13:49:50 2018

@author: nitesh
"""

class logs():

    def fileHandler(self,logging,log_file_path=None):
        """

        :param logging: Import of logging library
        :param log_file_path: log file save path
        :return:file_handler: Provides the log file handler where logs can be done
        """
        if log_file_path != None:
            formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(module)s:%(funcName)s:%(message)s')
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(formatter)

            return file_handler