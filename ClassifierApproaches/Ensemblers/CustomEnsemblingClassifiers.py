#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 11:39:36 2017

@author: nitesh
"""
import pandas as pd
class CustomEnsemblingClassifierSegments():
    """

    """
    def model_accuracy_store(self,accuracy_df=None,classifier_name=None,vec_name=None,model_name=None,prediction_perc=None):
        """

        :param accuracy_df: dataframe in that needs to store the accuracy of each model
        :param classifier_name: classifier name which is used for the model
        :param vec_name: vectorizer used inside the model
        :param model_name: model name that is generated
        :param prediction_perc: percentage of accuracy fetched by the model
        :return: dataframe that add  the percentage accuracy of the model
        """
        columns = ['classifier_name', 'vectorizer', 'model_name', 'prediction_accuracy_percentage']
        accuracy_df = accuracy_df.copy(deep=True)
        accuracy_df = accuracy_df.append(
            pd.DataFrame([[classifier_name, vec_name, model_name, prediction_perc]], columns=columns),
            ignore_index=True)
        return accuracy_df

    def get_mean_by_df(model_acc_df):
        """
        :param df: takes the data frame to find the mean value of result
        """
        mean_df = pd.DataFrame()
        #    temp1=df.groupby(['Classifier_Name']).agg({'Result':['mean']})
        mean_df = model_acc_df.groupby(['classifier_name']).mean()
        return mean_df

    def model_ensembling_result(self, accuracy_model_df=None,predicted_result_dict=None):
        """

        :return:
        """
        average_empty = list()
        predict_result_by_models = list()
        for i, ro in accuracy_model_df.iterrows():
            average_empty.append(0)
            predict_result_by_models(0)

        accuracy_model_df['average'] = average_empty
        accuracy_model_df['predicted_result'] = predict_result_by_models
        mean_df = self.get_mean_by_df(accuracy_model_df)
        for index, row in mean_df.iterrows():
            accuracy_model_df.loc[accuracy_model_df['classifier_name'] == row.name, 'average'] = row['prediction_accuracy']

        for key, value in predicted_result_dict.items():
            # df.loc[df.b <= 0, 'b'] = 0
            accuracy_model_df.loc[accuracy_model_df['model_name'] == key, 'predicted_result'] = value

        ma = accuracy_model_df.groupby(["classifier_name", "average", "predicted_result"])[
            "predicted_result"].count().reset_index(name="count")
        df1 = ma.sort_values('count', ascending=False).groupby('classifier_name').head(2)
        df1['score'] = df1['average'] * df1['count']
        top_df = df1.sort_values('score', ascending=False).head(2)
        return top_df,accuracy_model_df


