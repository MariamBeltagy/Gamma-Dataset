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