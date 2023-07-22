#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_svmlight_file
import sys
import numpy as np
import time
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
#from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, chi2
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint

# Load preprocessed data
df_morgan_filtervar=pd.read_csv('Fingerprint_filtervar.csv',index_col=0)
df_normalized_f_filtervar=pd.read_csv('normalized_filtervar.csv',index_col=0)
label=pd.read_csv('label.csv',index_col=0)
# Concatenate the features from two dataframes
X=pd.concat([df_morgan_filtervar,df_normalized_f_filtervar],axis=1)
# Scale the features using StandardScaler
scale = StandardScaler()
scaledX = scale.fit_transform(X)

# Set the target labels
y=label  # Labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaledX, y, test_size=0.2, random_state=12)
# Initialize the Random Forest classifier with custom class weights
rf = RandomForestClassifier(class_weight= {0: 0.039473684210526314, 1: 0.9605263157894737})
paramGrid_rf={"max_depth": [ int(x) for x in np.linspace( 10, 110, 11)]+[None],
              "n_estimators":[ int(x) for x in np.linspace( 200, 2000, 10)],
              "max_features": [ None, 0.5, "sqrt", "log2"],
               "min_samples_split": [ 2, 5, 10],
               'bootstrap':[True,False],
               'min_samples_leaf':randint(1,4),
              }

# initialize the random 3-fold CV search of hyperparameters
rf_Grid = RandomizedSearchCV(estimator=rf, param_distributions=paramGrid_rf, n_iter=100, cv = 3, verbose=2, n_jobs = 4)

# fit to the training data
rf_Grid.fit(X_train, y_train)
# Get the best hyperparameters found by RandomizedSearchCV
rf_Grid.best_params_
print (f'Train Accuracy - : {rf_Grid.score(X_train,y_train):.3f}')
print (f'Test Accuracy - : {rf_Grid.score(X_test,y_test):.3f}')
# Save the best hyperparameters using pickle
import pickle
with open('rf_tuning.pickle','wb') as f:
    pickle.dump(rf_Grid.best_params_,f)
# Load the best hyperparameters back using pickle   
with open('rf_tuning.pickle','rb') as f:
    rf_Grid.best_params_=pickle.load(f)

