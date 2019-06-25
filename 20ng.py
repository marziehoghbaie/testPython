# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 12:01:24 2019

@author: Novin Pendar

classification 
dataset 20ng
l2 normalization code
//
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
normalizer.fit(x_train)
x_train = normalizer.transform(x_train)
x_test = normalizer.transform(x_test)
//
"""

# convert docs to vectors of words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
x_train_file = open("scikit_learn_data/my text datasets/20ng/20ng-train-Stemmed.txt")
y_train_file = open("scikit_learn_data/my text datasets/20ng/20ng-train-label.txt")
x_test_file = open("scikit_learn_data/my text datasets/20ng/20ng-test-Stemmed.txt")
y_test_file = open("scikit_learn_data/my text datasets/20ng/20ng-test-label.txt")
x_train = x_train_file.readlines()
y_train = y_train_file.readlines()
x_test = x_test_file.readlines()
y_test = y_test_file.readlines()
x_train_file.close()
y_train_file.close()
x_test_file.close()
y_test_file.close()
# 
vectorizer = CountVectorizer()
# vectorizer = TfidfVectorizer()
# vectorizer = CountVectorizer(binary=True)
x_train=vectorizer.fit_transform(x_train)
x_test=vectorizer.transform(x_test)
