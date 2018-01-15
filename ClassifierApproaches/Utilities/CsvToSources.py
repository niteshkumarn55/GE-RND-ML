#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 14:00:15 2017

@author: nitesh
"""
from Preprocessing.DataMassaging import DataPreprocessing
import pandas as pd
import os
import codecs

class CsvToFile():
    """

    """
    def convert_to_files(self,path):
        """

        :param path:
        :return:
        """
        os.getcwd()
        file_source_path = os.getcwd()+r"/fintech"
        print("The file saving path is ",file_source_path)
        df = pd.read_csv(path)


        df = DataPreprocessing().get_text_without_stem(df=df)
        print(len(df))
        for index, row in df.iterrows():
            category = str(row['final_level_category'])
            category_file_path = file_source_path+"/"+category
            print("the file path is ",category_file_path)
            if not os.path.exists(category_file_path):
                os.makedirs(category_file_path)
            print("The id of the file is ", row['id'])
            file_path = category_file_path+"/"+str(row['id'])+".txt"

            print(file_path)
            file = codecs.open(file_path, 'w+', encoding='utf8')
            content = str(row['processed_text'])
            file.write(content)




if __name__ == '__main__':
    path = r"/Users/nitesh/OneDrive/Work/GE_Python_Workspace/ClassifierApproaches/FinTech_Nitesh_Data.csv"
    CsvToFile().convert_to_files(path)



