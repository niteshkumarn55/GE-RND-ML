#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 16:44:36 2017

@author: nitesh
"""
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer,CountVectorizer,TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from Utilities.AnalysisCalc import PredictionPrecentageCalc
import pickle
import pandas as pd
import os

class MultinomialClassifierSegment():

    def get_classified_vectorizer(self,vectorizer=None, classifier=None, classifier_name=None, X_train=None,y_train=None):
        """
        """
        #X_train = vectorizer.fit_transform(train_data['processed_text'].tolist())
        #X_test = vectorizer.transform(test_data['processed_text'].tolist())

        vec_clf = Pipeline([('vec', vectorizer),('tfidf', TfidfTransformer()), (classifier_name, classifier)])
        vec_clf = vec_clf.fit(X_train,y_train)

        return vec_clf



    def multinomail_classifier(self,X_train=None, y_train=None, doc_vectorizer=None, model_no=None,model_path=None):
        """
        This does the multinomaila classification
        """

        """Building Naive bayes Multinomial classifier model"""
        path = model_path+"Model"+str(model_no)
        if not os.path.exists(path):
            os.makedirs(path)
        vec_clf = self.get_classified_vectorizer(vectorizer=doc_vectorizer,classifier=MultinomialNB(), classifier_name='MultinomialNB',X_train=X_train,y_train=y_train)
        save_classifier = open(model_path+"Model"+str(model_no)+"/MultinomialClassifierTfidf"+str(model_no)+".pickle", "wb")
        pickle.dump(vec_clf,save_classifier)
        save_classifier.close()

    def load_classifier_model(self,model_no=None,model_path=None):
        """
        """
        pickle_in = open(model_path+"Model"+str(model_no)+"/MultinomialClassifierTfidf"+str(model_no)+".pickle","rb")
        model = pickle.load(pickle_in)
        pickle_in.close()
        return model

    def predict_from_model(self,df=None,model_no=None,model_path=None,X_test=None, y_test=None):
        """
        """
        predicted_model_name = "multinomial_model_" + str(model_no)
        predicted_model_accuracy_percentage = "multinomial_model_" + str(model_no) + "_accuracy"

        model = self.load_classifier_model(model_no=model_no, model_path=model_path)
        predicted_test_result = model.predict(X_test)


        df[predicted_model_name] = predicted_test_result
        prediction_accuracy = PredictionPrecentageCalc().predicted_model_result(predicted_data=predicted_test_result, y_test=y_test)
        df[predicted_model_accuracy_percentage] = prediction_accuracy

        print("Without cross validation Model name is multinomial Classifier : ", prediction_accuracy)
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
        df = df.append({'classifier':'Multinomail NB','accuracy':test_accuracy_result}, ignore_index=True)
        print("The Multinomial NB accuracy score of the test dataset is ", test_accuracy_result)

        return df
