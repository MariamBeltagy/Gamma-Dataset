# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 16:21:38 2019

@author: Aghapy
"""

# Import Used Libraries.
import csv
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as cs
from scipy.misc import imshow
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn import tree
import graphviz 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, precision_score, recall_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier


# Number of threads.
num_jobs = 9
# Reads a magic csv file.
def read_magic_data(file_directory):
  # Open up the file.
  f = open(file_directory)
  # Read line containing column names.
  headers = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "G", "classHG"]
  # Read data as matrix of strings.
  data = np.loadtxt(f, dtype = np.dtype(str),delimiter = ',')
  # Parse numbers as floats.
  x = data[:,:10].astype(np.dtype(float))
  # Transform categorical data classes to numbers.
  y = data[:,10]
  b, c = np.unique(y, return_inverse=True)
  # Make our data structured again.
  data[:,10] = c
  data[:,:10] = x
  data = data.astype(np.dtype(float))
  return data, headers, b



def print_stats(estimator, X_train, y_train, X_test, y_test):
  y_pred_train = estimator.predict(X_train)
  y_pred_test = estimator.predict(X_test)
  print('Train Set Accuracy : ', accuracy_score(y_train, y_pred_train))
  print('Train Set Precision : ', precision_score(y_train, y_pred_train))
  print('Train Set Recall : ', recall_score(y_train, y_pred_train))
  print('Train F-Score for each class : ', f1_score(y_train, y_pred_train, average=None))
  print('Train Mean F-Score for both classes : ',f1_score(y_train, y_pred_train, average='macro'))
  print('Train Confusion Matrix : ', confusion_matrix(y_train, y_pred_train))
  print('----------------------------------------------------------------------')
  print('Test Set Accuracy : ', accuracy_score(y_test, y_pred_test))
  print('Test Set Precision : ', precision_score(y_test, y_pred_test))
  print('Test Set Recall : ', recall_score(y_test, y_pred_test))
  print('Test F-Score for each class : ', f1_score(y_test, y_pred_test, average=None))
  print('Test Mean F-Score for both classes : ',f1_score(y_test, y_pred_test, average='macro'))
  print('Test Confusion Matrix : ', confusion_matrix(y_test, y_pred_test))
  print('----------------------------------------------------------------------')



#KNN
def kNearestNeighborsFunction(X_train, y_label_train, X_test, y_label_test, start_n, end_n):
  parameters = {'n_neighbors': list(range(start_n, end_n))}
  KNNC =  KNeighborsClassifier(n_jobs = num_jobs)
  gKnn = GridSearchCV(KNNC, parameters,scoring = 'f1_macro', cv = 5, n_jobs = num_jobs)
  gKnn.fit(X_train, y_label_train)
  print('Best N found at ', gKnn.best_params_)
  print('Best Mean F-Score ', gKnn.best_score_)
  ns = gKnn.cv_results_['param_n_neighbors']
  ns_score = gKnn.cv_results_['mean_test_score']
  plt.title('Mean F-Score with different neighbors')
  plt.plot(ns, ns_score)
  plt.xlabel('# of neighbours')
  plt.ylabel('Validation Mean F-Score')
  plt.show()
  print_stats(gKnn, X_train, y_label_train, X_test, y_label_test)







