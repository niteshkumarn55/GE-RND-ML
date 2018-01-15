#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 16:51:36 2017

@author: nitesh
"""

from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer,CountVectorizer,TfidfTransformer
from Utilities.AnalysisCalc import PredictionPrecentageCalc
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import pickle
import pandas as pd
import os

class DecisionTreeSegment():

    def get_classified_vectorizer(self,vectorizer=None, classifier=None, classifier_name=None, X_train=None,y_train=None):
        """

        :param vectorizer:
        :param classifier:
        :param classifier_name:
        :param X_train:
        :param y_train:
        :return:
        """
        #X_train = vectorizer.fit_transform(train_data['processed_text'].tolist())
        #X_test = vectorizer.transform(test_data['processed_text'].tolist())

        vec_clf = Pipeline([('vec', vectorizer),('tfidf', TfidfTransformer()), (classifier_name, classifier)])
        vec_clf = vec_clf.fit(X_train,y_train)

        return vec_clf

    def decision_tree_classifier(self,X_train=None, y_train=None, doc_vectorizer=None, model_no=None,model_path=None):
        """
        This does the decision tree classification
        :param X_train:
        :param y_train:
        :param doc_vectorizer:
        :param model_no:
        :param model_path:
        :return:
        """

        """Building Naive bayes DecisionTree Classifier model"""
        path = model_path+"Model"+str(model_no)
        if not os.path.exists(path):
            os.makedirs(path)

        dt = DecisionTreeClassifier(max_depth=7,criterion="gini")
        vec_clf = self.get_classified_vectorizer(vectorizer=doc_vectorizer, classifier=dt,
                                                 classifier_name='DecisionTree', X_train=X_train, y_train=y_train)
        save_classifier = open(model_path+"Model"+str(model_no)+"/DecisionTreeClassifierTfidf"+str(model_no)+".pickle", "wb")
        pickle.dump(vec_clf, save_classifier)
        save_classifier.close()
        return vec_clf

    def load_classifier_model(self,model_no=None,model_path=None):
        """

        :param model_no:
        :param model_path:
        :return:
        """
        pickle_in = open(model_path+"Model"+str(model_no)+"/DecisionTreeClassifierTfidf"+str(model_no)+".pickle","rb")
        model = pickle.load(pickle_in)
        pickle_in.close()
        return model

    def predict_from_model(self,df=None, model_no=None,model_path=None,X_test=None, y_test=None,X_train=None, y_train=None):
        """

        :param self:
        :param model_no:
        :param model_path:
        :param X_test:
        :param y_test:
        :return:
        """
        predicted_model_name = "decision_model_" + str(model_no)
        predicted_model_accuracy_percentage = "decision_model_" + str(model_no) + "_accuracy"

        model = self.load_classifier_model(model_no=model_no, model_path=model_path)
        predicted_test_result = model.predict(X_test)
        #Testing the score
        print("The decision T accuracy score of the test dataset is ",model.score(X_test,y_test))
        print("The decision T accuracy score of the training dataset is ", model.score(X_train,y_train))

        df[predicted_model_name] = predicted_test_result
        prediction_accuracy = PredictionPrecentageCalc().predicted_model_result(predicted_data=predicted_test_result,y_test=y_test)
        df[predicted_model_accuracy_percentage] = prediction_accuracy

        print("Without cross validation Model name is Decision Tree: ", prediction_accuracy)
        return df

    def classifier_predicted_for_unseen(self,model_vec_clf=None,X_test=None):
        """

        :param model_vec_clf:
        :param X_test:
        :return:
        """
        predicted_result = model_vec_clf.predict(X_test)
        return predicted_result

    def classifier_accuracy(self, model_vec_clf=None, X_test=None, y_test=None):
        """

        :param self:
        :param model_no:
        :param model_path:
        :param X_test:
        :param y_test:
        :return:
        """
        df = pd.DataFrame(columns=['classifier','accuracy'])
        #Testing the score
        test_accuracy_result = model_vec_clf.score(X_test,y_test)*100
        df = df.append({'classifier':'Decision Tree','accuracy':test_accuracy_result}, ignore_index=True)
        print("The decision T accuracy score of the test dataset is ", test_accuracy_result)

        return df,test_accuracy_result