#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 16:14:36 2017

@author: nitesh
"""
from Preprocessing.DataMassaging import DataPreprocessing
from Utilities.SplitDatasets import CustomSplits
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from Classifiers.Decisiontree import DecisionTreeSegment
from Classifiers.MultinomialClassifer import MultinomialClassifierSegment
from Classifiers.KNNClassifier import KnnClassifierSegment
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from Utilities.AnalysisCalc import ConfusionMatrixCalc,AccuracyAnalysis
from MatplotCharts.Charts import GEPlots
import numpy as np
import pickle
import os

class ModelAccuracy():
    """

    """


    _technology_column = "technology_segment_2"
    df = pd.read_csv(r"/Users/nitesh/OneDrive/Work/GE_Python_Workspace/ClassifierApproaches/Excel Documents/"
                     r"Fintech/Fintech_Digital_Banking_Training_Data.csv")
    df, X,Y= DataPreprocessing().get_complete_text(df=df,technology_column=_technology_column)
    def decision_tree_model_accuracy(self):
        """

        :return:
        """
        decision_tree_model_save_dir = r"/Users/nitesh/OneDrive/Work/GE_Python_Workspace/ClassifierApproaches/" \
                          r"Model_Accuracy_csv/Fintech/Digital_banking/decision/"
        doc_vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                         stop_words='english')
        dtc = DecisionTreeClassifier(max_depth=7, criterion="gini")
        cmc = ConfusionMatrixCalc()
        model_accuracy_df = pd.DataFrame(columns=['index','classifier','accuracy'])

        for i in range(200):
            dts = DecisionTreeSegment()
            labels = self.Y[self._technology_column]
            X_train, X_test, Y_train, Y_test = CustomSplits().random_X_and_Y(X=self.X, Y=labels)
            GEPlots().count_of_each_categories(y_train=Y_train,y_test=Y_test)
            np_x_train, np_x_test = X_train['processed_text'], \
                                                           X_test['processed_text']
            vec_clf = dts.get_classified_vectorizer(vectorizer=doc_vectorizer, classifier=dtc,
                                                     classifier_name='DecisionTree', X_train=np_x_train,
                                                    y_train=Y_train)
            temp_df,test_accuracy_score = dts.classifier_accuracy(model_vec_clf=vec_clf,X_test=np_x_test,y_test=Y_test)
            x_predict = dts.classifier_predicted_for_unseen(model_vec_clf=vec_clf, X_test=np_x_test)
            # print(type(x_predict))
            # print(len(x_predict))
            pred_unique, pred_count = np.unique(x_predict,return_counts=True)
            # pred_dict = dict(zip(pred_unique, pred_count))

            y_test_temp = Y_test.as_matrix()
            # print(type(y_test_temp))
            # print(len(y_test_temp))

            y_unique_labels, y_count = np.unique(y_test_temp,return_counts=True)
            # y_dict = dict(zip(y_unique_labels, y_count))
            # print(y_unique_labels.tolist())
            # print(y_unique_labels)
            # print(count)
            AccuracyAnalysis().total_pred_and_expected_count(predicted_unique_labels=pred_unique.tolist(),
                                                             y_unique_labels=y_unique_labels.tolist(),
                                                             predicted_counts=pred_count.tolist(),
                                                             y_counts=y_count.tolist())

            model_accuracy_df = model_accuracy_df.append({'index': i+1, 'classifier':temp_df['classifier'].iloc[0],
                                                        'accuracy':temp_df['accuracy'].iloc[0]}, ignore_index=True)
            save_img_path = "decision_tree_plot/" + str(i) + ".png"
            cmc.confusion_matrix_calc(y_test=y_test_temp, y_pred=x_predict, class_names=y_unique_labels,
                                      save_image=save_img_path)
            if test_accuracy_score > 69:
                if not os.path.exists(decision_tree_model_save_dir):
                    os.makedirs(decision_tree_model_save_dir)

                model_name = "decision_tree_model"+str(i)
                self.save_model(model_path=decision_tree_model_save_dir,model_name=model_name,model=vec_clf)

        decision_accuracy_path = r"/Users/nitesh/OneDrive/Work/GE_Python_Workspace/ClassifierApproaches/" \
                                 r"Model_Accuracy_csv/Fintech/Digital_banking/decision.csv"
        model_accuracy_df.to_csv(decision_accuracy_path)


    def knn_model_accuracy(self):
        """

        :return:
        """
        doc_vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                         stop_words='english')
        kc = KNeighborsClassifier()
        cmc = ConfusionMatrixCalc()
        model_accuracy_df = pd.DataFrame(columns=['index', 'classifier', 'accuracy'])

        for i in range(200):
            kcs = KnnClassifierSegment()
            X_train, X_test, Y_train, Y_test = CustomSplits().random_X_and_Y(X=self.X, Y=self.Y)
            np_x_train, np_x_test, np_y_train, np_y_test = X_train['processed_text'],\
                                                            X_test['processed_text'], Y_train[self._technology_column],\
                                                            Y_test[self._technology_column]
            vec_clf = kcs.get_classified_vectorizer(vectorizer=doc_vectorizer, classifier=kc,
                                                    classifier_name='knn', X_train=np_x_train, y_train=np_y_train)
            temp_df = kcs.classifier_accuracy(model_vec_clf=vec_clf, X_test=np_x_test, y_test=np_y_test)
            x_predict = kcs.classifier_predicted_for_unseen(model_vec_clf=vec_clf,X_test=X_test)
            model_accuracy_df = model_accuracy_df.append({'index': i + 1, 'classifier': temp_df['classifier'].iloc[0],
                                                          'accuracy': temp_df['accuracy'].iloc[0]}, ignore_index=True)
            save_img_path = "knn_plot/"+str(i)+".png"
            cmc.confusion_matrix_calc(y_test=Y_test,y_pred=x_predict,class_names=np_y_train.unique(),save_image=save_img_path)

        knn_accuracy_path = r"/Users/nitesh/OneDrive/Work/GE_Python_Workspace/ClassifierApproaches/" \
                            r"Model_Accuracy_csv/Fintech/Digital_banking/knn.csv"
        model_accuracy_df.to_csv(knn_accuracy_path)


    def multinomial_model_accuracy(self):
        """

        :return:
        """
        doc_vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                         stop_words='english')
        MNB = MultinomialNB()
        cmc = ConfusionMatrixCalc()
        model_accuracy_df = pd.DataFrame(columns=['index', 'classifier', 'accuracy'])

        for i in range(200):
            mcs = MultinomialClassifierSegment()
            X_train, X_test, Y_train, Y_test = CustomSplits().random_X_and_Y(X=self.X, Y=self.Y)
            np_x_train, np_x_test, np_y_train, np_y_test = X_train['processed_text'], \
                                                           X_test['processed_text'], Y_train[self._technology_column], \
                                                           Y_test[self._technology_column]
            vec_clf = mcs.get_classified_vectorizer(vectorizer=doc_vectorizer, classifier=MNB,
                                                    classifier_name='NB', X_train=np_x_train, y_train=np_y_train)
            temp_df = mcs.classifier_accuracy(model_vec_clf=vec_clf, X_test=np_x_test, y_test=np_y_test)
            x_predict = mcs.classifier_predicted_for_unseen(model_vec_clf=vec_clf, X_test=X_test)
            model_accuracy_df = model_accuracy_df.append({'index': i + 1, 'classifier': temp_df['classifier'].iloc[0],
                                                          'accuracy': temp_df['accuracy'].iloc[0]}, ignore_index=True)
            save_img_path = "knn_plot/" + str(i) + ".png"
            cmc.confusion_matrix_calc(y_test=Y_test, y_pred=x_predict, class_names=np_y_train.unique(),
                                      save_image=save_img_path)
        mnb_accuracy_path = r"/Users/nitesh/OneDrive/Work/GE_Python_Workspace/ClassifierApproaches/" \
                            r"Model_Accuracy_csv/Fintech/Digital_banking/multinomialNB.csv"
        model_accuracy_df.to_csv(mnb_accuracy_path)


    def save_model(self,model_path=None,model_name=None,model=None):
        """

        :param model_path:
        :param model:
        :return:
        """

        save_classifier = open(model_path + model_name + ".pickle", "wb")
        pickle.dump(model, save_classifier)
        save_classifier.close()



if __name__ == '__main__':
    mc = ModelAccuracy()
    mc.decision_tree_model_accuracy()
    # print("*"*10," Knn ")
    # mc.knn_model_accuracy()
    # print("*" * 10, " NB ")
    # mc.multinomial_model_accuracy()