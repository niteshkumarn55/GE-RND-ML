#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 18:25:19 2017

@author: nitesh
"""

import matplotlib.pyplot as plt
import itertools
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import table
import pandas as pd

from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix

class PredictionPrecentageCalc():
    """

    """
    def percentage_correct_prediction(self,worng_predict, total_data):
        percentage = ((total_data - worng_predict)/total_data)*100
        return percentage


    def predicted_model_result(self,predicted_data=None,y_test=None):
        """

        :param predicted_data:
        :param y_test:
        :return:
        """
        count_of_wrong_predict = 0

        for index,value in enumerate(predicted_data):
            expected_value = y_test.iloc[index] # or this will also work  : test_data.iloc[index:1]
            if value != expected_value:
                count_of_wrong_predict +=1
        return self.percentage_correct_prediction(count_of_wrong_predict,len(y_test))

class AccuracyAnalysis():
    """

    """
    def total_pred_and_expected_count(self,predicted_unique_labels=None,y_unique_labels=None,predicted_counts=None,
                                      y_counts=None,y_dict=None,pred_dict=None):
        """

        :param predicted_unique_labels:
        :param y_unique_labels:
        :param predicted_counts:
        :param y_counts:
        :return:
        """
        data = {'index':y_unique_labels,'expected_result':y_counts,'obtained_results':predicted_counts}
        # df = pd.DataFrame([(y_unique_labels, y_counts, predicted_counts)], columns=columns)
        df = pd.DataFrame(data=data)
        # Plotting the a graph of test data split
        fig, ax = plt.subplots(1, 1)
        table(ax, df,
              loc='upper right', colWidths=[0.2, 0.2, 0.2])
        df.plot(ax=ax)

        plt.savefig(r"/Users/nitesh/OneDrive/Work/GE_Python_Workspace/ClassifierApproaches/Model_Accuracy_csv/"
                    r"Fintech/Digital_banking/category_count_accuracy.png")
        print(df)


class ConfusionMatrixCalc():
    """

    """

    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


    def confusion_matrix_calc(self,y_test=None,y_pred=None,class_names=None,save_image=None):

        cnf_matrix = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=class_names,
                              title='Confusion matrix, without normalization')
        plt_img_non_normalized = "non_normalized_" + str(save_image) #Name of the non-normalized image to be stored
        if save_image != None:
            plt.savefig(r"/Users/nitesh/OneDrive/Work/GE_Python_Workspace/ClassifierApproaches/Model_Accuracy_csv/Fintech/Digital_banking/"+plt_img_non_normalized)


        # Plot normalized confusion matrix
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                              title='Normalized confusion matrix')
        plt_img_normalized = "normalized_" + str(save_image)  # Name of the normalized image to be stored
        if save_image != None:
            plt.savefig(r"/Users/nitesh/OneDrive/Work/GE_Python_Workspace/ClassifierApproaches/Model_Accuracy_csv/Fintech/Digital_banking/"+plt_img_normalized)

        if save_image == None:
            plt.show()

