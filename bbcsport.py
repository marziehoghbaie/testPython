# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 13:26:03 2019

@author: Novin Pendar
classification 
dataset bbcsport
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
from sklearn.model_selection import train_test_split
x_file = open("scikit_learn_data/my text datasets/bbcsport/bbcsport-Stemmed.txt")
y_file = open("scikit_learn_data/my text datasets/bbcsport/bbcsport-label.txt")
x = x_file.readlines()
y = y_file.readlines()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_file.close()
y_file.close()
vectorizer = TfidfVectorizer()
# vectorizer = TfidfVectorizer()
# vectorizer = CountVectorizer(binary=True)
x_train=vectorizer.fit_transform(x_train)
x_test=vectorizer.transform(x_test)