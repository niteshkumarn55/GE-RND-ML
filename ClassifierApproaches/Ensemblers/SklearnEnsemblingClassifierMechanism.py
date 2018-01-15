#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:19:32 2017

@author: nitesh
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer,CountVectorizer,TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn. ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from Ensemblers.Ensembler import EnsembleClassifier
from sklearn.pipeline import Pipeline
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from Utilities.AnalysisCalc import PredictionPrecentageCalc
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from Utilities.AnalysisCalc import ConfusionMatrixCalc

class EnsemblingMechanism():
    """

    """

    def bagging_classifier(self, X_train=None,y_train =None, X_test=None,y_test=None,classifier=None,n_estimator=None):
        """

        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
        :param classifier:
        :param n_estimator:
        :return:
        """
        bg = BaggingClassifier(classifier, max_samples=0.5,max_features=1.0,n_estimators=n_estimator)
        bg.fit(X_train,y_train)
        score = bg.score(X_test,y_test)
        print("The bagging classifier prediction score with train dataset ",score)
        bg.predict()
        return score, bg

    def bagging_classifier_prediction(self,bg=None, X_test=None,y_test=None):
        """

        :param bg:
        :param X_test:
        :param y_test:
        :return:
        """
        predicted_result = bg.predict(X_test)
        PredictionPrecentageCalc().predicted_model_result(predicted_data=predicted_result,y_test=y_test)
        return predicted_result



    def ada_boosting_classifier(self,X_train=None,y_train=None,X_test=None,y_test=None,classifier=None,n_estimator=None):
        """

        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
        :param classifier:
        :param n_estimator:
        :return:
        """
        # Boosting - Ada Boost

        adb = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=n_estimator, learning_rate=1)
        adb.fit(X_train, y_train)
        score = adb.score(X_test,y_test)
        print("The ada boosting prediction score with train dataset ", score)
        return score, adb

    def ada_boosting_classifier(self,adb=None, X_test=None,y_test=None):
        """

        :param adb: ada boosting object
        :param X_test: test dataset
        :param y_test: test labels/categories set
        :return:
        """
        predicted_result = adb.predict(X_test)
        PredictionPrecentageCalc().predicted_model_result(predicted_data=predicted_result, y_test=y_test)
        return predicted_result

    def ensembling_voting_classifier(self,X_train=None,y_train=None,X_test=None,y_test=None,estimator_classifier=None,
                                     vectorizer=None,weight=None):
        """

        :param X_train: training dataset
        :param y_train: training label/categories set
        :param X_test: test dataset
        :param y_test: test labels/categories set
        :param estimator_classifier:  list of tuple example [('decision_tree',dt),('svm',svm)]. Tuple providing the name
                                      and the classifier object respectively
        :param weight: list of weights [1,1,3], weights for all the classifiers
        :return:
        """

        evc = VotingClassifier(estimators=estimator_classifier,voting='hard',weights=weight)
        X_train_tfidf = vectorizer.fit_transform(X_train)

        X_test_tfidf = vectorizer.transform(X_test)
        vec_clf = Pipeline([('vec', vectorizer), ('tfidf', TfidfTransformer()), ('evc', evc)])

        evc.fit(X_train_tfidf,y_train)
        vec_clf.fit(X_train,y_train)
        score_piped_training_data = vec_clf.score(X_train,y_train)
        score_piped_test_data = vec_clf.score(X_test,y_test)
        score_training_data = evc.score(X_train_tfidf,y_train)
        score_test_data = evc.score(X_test_tfidf,y_test)
        predicted = evc.predict(X_test_tfidf)
        predicted2 = vec_clf.predict(X_train)
        print("training data ",predicted2)
        print("training data without pipe",predicted)
        # print(y_test['ml_segment_data1_ap4_t2'])
        print("The score of ensembled custom piped vectorizer clf for the training data ", score_piped_training_data)
        print("The score of ensembled vectorizer clf for the training data ", score_training_data)
        print("The score of ensembled custom piped vectorizer clf for the test data ", score_piped_test_data)
        print("The score of ensembled vectorizer clf for the test data ", score_test_data)
        print("The score of test data is ", np.mean(predicted == y_test))
        ConfusionMatrixCalc().confusion_matrix_calc(y_test=y_test,y_pred=predicted,class_names=y_train.unique())
        return vec_clf

    def ensembling_voting_classifier_prediction(self,evc=None, X_test=None,y_test=None):
        """

        :param evc:
        :param X_test:
        :param y_test:
        :return:
        """
        predicted_result = evc.predict(X_test)
        precentage =PredictionPrecentageCalc().predicted_model_result(predicted_data=predicted_result, y_test=y_test)
        print("The voting ensembling percentage ", precentage)
        print("The voting ensembling score ",evc.score(X_test,y_test))
        return predicted_result

    def ensembling_weight_classification_prediction(self,X_train=None, Y_train=None,X_test=None,Y_test=None,vectorizer=None):
        """

        :param ewc:
        :param X_train:
        :param Y_train:
        :param X_test:
        :param Y_test:
        :return:
        """


        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        clf1 = DecisionTreeClassifier(max_depth=5)
        clf2 = RandomForestClassifier()
        clf3 = RandomForestClassifier(criterion='entropy', max_depth=4)
        # clf3 = SVC()
        eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3], weights=[1, 1, 1])
        eclf.fit(X_train_tfidf,Y_train)
        print(eclf.score(X_test_tfidf,Y_test))

        for clf, label in zip([clf1, clf2, clf3, eclf],
                              ['Logistic Regression', 'Random Forest', 'random_forest_entropy', 'Ensemble']):
            clf.fit(X_train_tfidf,Y_train)
            print(clf.score(X_train_tfidf,Y_train))
            # print("the test data ",clf.score(X_test_tfidf,Y_test))
            scores = model_selection.cross_val_score(clf, X_train_tfidf, Y_train, cv=5, scoring='accuracy')
            print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))



