#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 13:46:19 2018

@author: nitesh
"""
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation,digits
import re
import os
import logging
from LoggingDetails.LogFileHandler import logs
from LoggingDetails.LogPathConstant import LogFilePathContants,LogFiles

log_file = os.path.join(LogFilePathContants._BASE_LOG_FILE,
                        LogFiles._DATA_MASSAGE_LOG_FILE)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logs().fileHandler(logging,log_file_path=log_file)
logger.addHandler(file_handler)

class ProcessingData():

    _stopwords = set(stopwords.words('english') + list(punctuation) + list(digits))
    _stemmer = SnowballStemmer('english')

    def nltk_preprocessing(self, content):
        """
        simple preprocessing step is taken care.
        1> converts the words to lowercase.
        2> removes the stopwords
        3> stems all the words

        :param content: Expects the raw doc
        :return preprocessed text
        """
        text = ""
        tokens = word_tokenize(content)
        for token in tokens:
            token = token.lower()
            if token not in self._stopwords:
                stemmed_token = self._stemmer.stem(token)
                if text == "":
                    text = stemmed_token
                else:
                    text = text + ' ' + stemmed_token
        return text

    def nltk_without_stem(self, content):
        """

        :param content:
        :return:
        """
        text = ""
        word_regex = re.compile(r"\w")
        tokens = word_tokenize(content)
        for token in tokens:
            token = token.lower()
            if token not in self._stopwords:
                if word_regex.match(token):
                    if text == "":
                        text = token
                    else:
                        text = text + ' ' + token
        return text

    def get_complete_text(self, df=None, technology_column=None):
        """
        Takes the specialties, short_desc, about us and concatinates and provides a complete contacatated result in the df
        :param df: Need the df which has the above mentioned columns
        :return dataframe
        """
        logger.info("Combining the short_description and about us")
        # df['complete_text'] = df[['specialties', 'specialties', 'short_description', 'about_us']].apply(
        #     lambda x: x.str.cat(sep=' '), axis=1)
        df['complete_text'] = df[['short_description', 'about_us']].apply(
            lambda x: x.str.cat(sep=' '), axis=1)
        df['processed_text'] = df['complete_text'].apply(ProcessingData().nltk_preprocessing)
        logger.info("df now contains the processed_text column with stemming of words")
        return df

    def get_text_without_stem(self, df=None):
        """
        Takes the specialties, short_desc, about us and concatinates and provides a complete contacatated result in the df
        :param df: Need the df which has the above mentioned columns
        :return dataframe
        """
        logger.info("Combining the short_description and about us without stemming the text")
        df['complete_text'] = df[['short_description', 'about_us']].apply(
            lambda x: x.str.cat(sep=' '), axis=1)
        df['processed_text'] = df['complete_text'].apply(ProcessingData().nltk_without_stem)
        logger.info("df now contains the processed_text column which is not stemmed")
        return df
