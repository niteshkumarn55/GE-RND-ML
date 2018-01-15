#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 15:12:36 2017

@author: nitesh
"""

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import pandas as pd
import re


class DataPreprocessing():


    _stopwords = set(stopwords.words('english') + list(punctuation))
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

    def nltk_without_stem(self,content):
        """

        :param content:
        :return:
        """
        text = ""
        word_regex = re.compile(r"\w")
        tokens = word_tokenize(content)
        for token in tokens:
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
        # df['complete_text'] = df[['specialties', 'specialties', 'short_description', 'about_us']].apply(
        #     lambda x: x.str.cat(sep=' '), axis=1)
        df['complete_text'] = df[['short_description', 'about_us']].apply(
            lambda x: x.str.cat(sep=' '), axis=1)
        df['processed_text'] = df['complete_text'].apply(DataPreprocessing().nltk_preprocessing)
        #    X = df.iloc['processed_text']
        #    Y = df.iloc[technology_column]
        X = pd.DataFrame()
        Y = pd.DataFrame()
        X['processed_text'] = df['processed_text']
        X['domain_name'] = df['domain_name']
        Y[technology_column] = df[technology_column]
        return df, X, Y


    def get_text_without_stem(self, df=None):
        """
        Takes the specialties, short_desc, about us and concatinates and provides a complete contacatated result in the df
        :param df: Need the df which has the above mentioned columns
        :return dataframe
        """
        df['complete_text'] = df[['specialties', 'specialties', 'short_description', 'about_us']].apply(
            lambda x: x.str.cat(sep=' '), axis=1)
        df['processed_text'] = df['complete_text'].apply(DataPreprocessing().nltk_without_stem)
        return df