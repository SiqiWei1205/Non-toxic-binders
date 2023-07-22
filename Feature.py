#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_svmlight_file
import sys
import numpy as np
import time
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, chi2
import seaborn as sns
from sklearn.preprocessing import StandardScaler

import pandas as pd

# Load preprocessed data
df_morgan_filtervar = pd.read_csv('Fingerprint_filtervar.csv', index_col=0)
df_normalized_f_filtervar = pd.read_csv('normalized_filtervar.csv', index_col=0)
label = pd.read_csv('label.csv')

# Combine features from fingerprint and normalized data
X = pd.concat([df_morgan_filtervar, df_normalized_f_filtervar], axis=1)

# Scale the features using Min-Max Scaling
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
X_scaler = min_max_scaler.fit_transform(X)
print("\nAfter min max Scaling :\n", X_scaler)

# Standardize the features
standardization = preprocessing.StandardScaler()
x_after_standardization = standardization.fit_transform(X)
print("\nAfter Standardisation :\n", x_after_standardization)

# Get the labels from the 'label' DataFrame
y = label.iloc[:, 1].values

# Create a new DataFrame 'data' by combining features and labels
data = X
data['label'] = y

# Calculate the correlation between features and labels and plot a horizontal bar chart
data.corr().loc['label'].plot(kind='barh')

# Delete the features that have weak correlation with the label
corr = abs(data.corr().loc['label'])
corr = corr[corr < 0.01]
cols_to_drop = corr.index.to_list()
data = data.drop(cols_to_drop, axis=1)

# Calculate the absolute correlation matrix for the 'data' DataFrame
cor_matrix = data.corr().abs()

# Extract the upper triangle of the correlation matrix
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))

# Find the features that have high correlation with each other (above 0.95)
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]

# Drop the features with high correlation
df1 = data.drop(data.columns[to_drop], axis=1)
print(df1.head())

# Save the 'data' DataFrame to a pickle file named 'rf_tuning.pickle'
import pickle
with open('rf_tuning.pickle', 'wb') as f:
    pickle.dump(data, f)

# Load the 'data' DataFrame from the pickle file 'rf_tuning.pickle'
with open('rf_tuning.pickle', 'rb') as f:
    data = pickle.load(f)
