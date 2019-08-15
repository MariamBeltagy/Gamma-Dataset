# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 16:21:38 2019
@author: Aghapy
"""

# Import Used Libraries.

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import random
import pandas as pd
import numpy as np
from openpyxl.utils import dataframe
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

import seaborn as sns
from math import sqrt
import matplotlib.pyplot as plt
import math
from matplotlib import colors as cs
from scipy.misc import imshow
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale
from sklearn import preprocessing, metrics
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn import tree
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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import GridSearchCV


magic = pd.read_csv("magic.data", sep=',')  # Reads a magic csv file.
magic.classHG = pd.get_dummies(magic['classHG'], drop_first=True)  # class H=1  G=0

x = magic.drop("classHG", axis=1)
y = magic["classHG"]

# Visualization
plt.title('line plot')  # line plot
plt.plot(x, y)
""""we can able to more clearly see the rate of change (slope)
 between individual data points. If the independent variable was nominal,
  you would almost certainly use a bar graph instead of a line graph."
"""
plt.savefig('lineplot.png')
plt.show()
#######################################
magic.hist()  # histogram
plt.savefig('histogram.png')
"""we realize that in the histogram of classHG the number of samples in class G is
much larger than class H (we will undersampe that later), also we can predict from the histogram of
the other features which one would be the best"""
########################################
plt.matshow(magic.corr())  # correlation matrix
""" the observable pattern is that all the variables highly correlated
 with each other the linear regression’s estimates will be unreliable"""
plt.title('correlation matrix')
plt.savefig('correlationMatrix.png')
plt.show()
########################################
magic.boxplot(grid=False) # box plot
plt.savefig('boxplot.png')
"""we can see that the distribution of values of some features is very small and 
large for other this will effect the features that will be used in classification  
"""
plt.show()
########################################

# dataset undersampling
classHG_count = magic.classHG.value_counts()
count_class_G, count_class_H = magic.classHG.value_counts()
# Divide by class
G_class = magic[magic['classHG'] == 0]
H_class = magic[magic['classHG'] == 1]
G_class_under = G_class.sample(count_class_H)
magic_under = pd.concat([G_class_under, H_class], axis=0)
x_under = magic_under.drop("classHG", axis=1)
y_under = magic_under["classHG"]
magic_under.hist()
plt.show()

#####################
# normlization of data (1,0)
scaler = MinMaxScaler()
scaler.fit(magic_under)
magic_norm = scaler.transform(magic_under)
np.savetxt("foo.csv", magic_norm, delimiter=",")
magic_norm = pd.read_csv("foo.csv", sep=',')
xx = magic_norm.iloc[:, 0:10]  # independent columns
xx.columns = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'
]
xx
yy = magic_norm.iloc[:, -1]  # target column

##Feature importance gives you a score for each feature of your data, the higher the score more important or relevant is the feature towards your output variable.
# apply SelectKBest class to extract  features
bestfeatures = SelectKBest(score_func=chi2, k=2)
fit = bestfeatures.fit(xx, yy)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(xx.columns)
# concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
# print(featureScores.nlargest(10,'Score'))

##############
model = ExtraTreesClassifier()
model.fit(xx, yy)
# print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
# plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=xx.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.savefig('bestfeatures.png')
plt.show()
#################

x_under = magic_under.drop(["classHG", "D", "E", "F", "G", "H", "J"], axis=1)

# split data into training and testing
x_train, x_test, y_train, y_test = train_test_split(x_under, y_under, test_size=0.3, random_state=1)


#############################


def print_stats(estimator):
    y_pred_train = estimator.predict(x_train)
    y_pred_test = estimator.predict(x_test)
    print('Train Set Accuracy : ', accuracy_score(y_train, y_pred_train))
    print('Train Set Precision : ', precision_score(y_train, y_pred_train))
    print('Train Set Recall : ', recall_score(y_train, y_pred_train))
    print('Train F-Score for each class : ', f1_score(y_train, y_pred_train, average=None))
    print('Train Mean F-Score for both classes : ', f1_score(y_train, y_pred_train, average='macro'))
    print('Train Confusion Matrix : ', confusion_matrix(y_train, y_pred_train))
    print('----------------------------------------------------------------------')
    print('Test Set Accuracy : ', accuracy_score(y_test, y_pred_test))
    print('Test Set Precision : ', precision_score(y_test, y_pred_test))
    print('Test Set Recall : ', recall_score(y_test, y_pred_test))
    print('Test F-Score for each class : ', f1_score(y_test, y_pred_test, average=None))
    print('Test Mean F-Score for both classes : ', f1_score(y_test, y_pred_test, average='macro'))
    print('Test Confusion Matrix : ', confusion_matrix(y_test, y_pred_test))
    print('----------------------------------------------------------------------')


# KNN
def kNearestNeighborsFunction():
    neighbors = list(range(1, 30, 2))
    # empty list that will hold cv scores
    cv_scores = []
    # perform 10-fold cross validation
    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())

    mse = [1 - x for x in cv_scores]
    optimal_k = neighbors[mse.index(min(mse))]
    n=optimal_k
    print('Best N found at ' , n)
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(x_train , y_train)
    print_stats(knn)


# logistic regression
def LR():
    grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"]}  # l1 lasso l2 ridge
    logreg = LogisticRegression()
    logreg_cv = GridSearchCV(logreg, grid, cv=10)
    logreg_cv.fit(x_train, y_train)
    print("tuned hpyerparameters :(best parameters) ", logreg_cv.best_params_)
    print_stats(logreg_cv)
    print("accuracy :", logreg_cv.best_score_)


# naive bayes
def NB():
    nb = GaussianNB()
    nb = nb.fit(x_train, y_train)
    print_stats(nb)

#Decision Tree
def DT():
    ct_gini = DecisionTreeClassifier()
    ct_entropy = DecisionTreeClassifier(criterion="entropy")
    ct_gini = ct_gini.fit(x_train, y_train)
    ct_entropy = ct_entropy.fit(x_train, y_train)
    print_stats(ct_entropy)
    print_stats(ct_gini)

#AdaBoost
def AB():
    param_grid = {
        'learning_rate': [.1, .2, .3, .4, .5],

        'n_estimators': [50, 100, 150, 200, 250]
    }

    classifier = AdaBoostClassifier()
    grid_Search = GridSearchCV(classifier, param_grid=param_grid)
    kk = grid_Search.fit(x_train, y_train)
    print_stats(kk)


AB()
