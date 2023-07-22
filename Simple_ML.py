#!/usr/bin/env python
# coding: utf-8

# load library
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_svmlight_file
import sys
import numpy as np
import time
import joblib
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_decision_regions, plot_confusion_matrix
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, chi2
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load data from CSV files
df_morgan_filtervar=pd.read_csv('Fingerprint_filtervar.csv',index_col=0)
df_normalized_f_filtervar=pd.read_csv('normalized_filtervar.csv',index_col=0)
label=pd.read_csv('label.csv')


# Random Forest

# Combine two datasets df_morgan_filtervar and df_normalized_f_filtervar into X
X = pd.concat([df_morgan_filtervar, df_normalized_f_filtervar], axis=1)

# Standardize the features in X using StandardScaler
scale = StandardScaler()
scaledX = scale.fit_transform(X)

# Store the labels in y
y = label

# PART 4: Handling the missing values
from sklearn import preprocessing

# MIN MAX SCALER
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
# Scaled feature
X_scaler = min_max_scaler.fit_transform(X)
print("\nAfter min max Scaling : \n", X_scaler)

# Standardisation
Standardisation = preprocessing.StandardScaler()
# Scaled feature
x_after_Standardisation = Standardisation.fit_transform(X)
print("\nAfter Standardisation : \n", x_after_Standardisation)

# Feature selection using VarianceThreshold
var_threshold = VarianceThreshold(threshold=0.05)
# Fit the data
var_threshold.fit(X)
# Transform the data by removing low-variance features
X_filtered = var_threshold.transform(X)

print(X_filtered)
print('*' * 10, "Separator", '*' * 10)

# shapes of data before transformed and after transformed
print("Earlier shape of data: ", X.shape)
print("Shape after transformation: ", var_threshold.transform(X).shape)
# Store the original data in variable data
data = X

# Add the label column to the data from y
data['label'] = y.iloc[:, 1].values

# Calculate the correlation between the label and each feature and plot the result
data.corr().loc['label'].plot(kind='barh')

# Delete the features without strong correlation with the label
corr = abs(data.corr().loc['label'])
corr = corr[corr < 0.01]
cols_to_drop = corr.index.to_list()
data = data.drop(cols_to_drop, axis=1)

# Calculate the correlation matrix for the remaining features and store the absolute values
cor_matrix = data.corr().abs()

# Create an upper triangular matrix of boolean values where True indicates a correlation value > 0.8
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))

# Find the columns to drop where any correlation value is greater than 0.8

to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]

# Drop the columns with high correlation
data = data.drop(to_drop, axis=1)

# Separate the labels (y) and the features (X) after dropping the irrelevant columns
y = data['label']
X = data.loc[:, data.columns != 'label']
X.reset_index(inplace=True)
X.drop("smiles", axis=1, inplace=True)

# Save the modified data to a CSV file named 'features_new.csv'
data.to_csv('features_new.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

print("Number transactions train dataset: ", len(X_train))
print("Number transactions test dataset: ", len(X_test))
print("Total number: ", len(X_train) + len(X_test))

# Create a Random Forest classifier with 100 estimators and fit it to the training data
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)


# Grid search for class weight
from sklearn.model_selection import GridSearchCV

# Define a range of weights from 0.05 to 0.95
weights = np.linspace(0.05, 0.95, 20)

# Create a GridSearchCV object for RandomForestClassifier with 'f1' scoring and 3-fold cross-validation
# The parameter grid consists of different class weights for class 0 and class 1
gsc = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid={
        'class_weight': [{0: x, 1: 1.0-x} for x in weights]
    },
    scoring='f1',
    cv=3
)

# Fit the GridSearchCV object to the data (X, y)
grid_result = gsc.fit(X, y)

# Load the GridSearchCV results from the 'gc_rf_weight.pickle' file using pickle
import pickle
with open('gc_rf_weight.pickle','rb') as f:
    grid_result=pickle.load(f)

# Define another range of weights from 0.01 to 0.05
weights = np.linspace(0.01, 0.05, 20)

# Create a new GridSearchCV object with the new weight range
gsc = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid={
        'class_weight': [{0: x, 1: 1.0-x} for x in weights]
    },
    scoring='f1',
    cv=3
)

# Fit the new GridSearchCV object to the data (X, y)
grid_result = gsc.fit(X, y)

# Load the GridSearchCV results from the 'gc_rf_weight_1.pickle' file using pickle
import pickle
with open('gc_rf_weight_1.pickle','rb') as f:
    grid_result=pickle.load(f)

# Print the best parameters found by the GridSearchCV
print("Best parameters : %s" % grid_result.best_params_)

# Plot the weights vs f1 score to visualize the performance of different class weights
dataz = pd.DataFrame({ 'score': grid_result.cv_results_['mean_test_score'],
                       'weight': weights })
dataz.plot(x='weight')

# Create a RandomForestClassifier with class weights {0: 0.05, 1: 0.95}
rf_w = RandomForestClassifier(class_weight= {0: 0.05, 1: 0.95})

# Fit the RandomForestClassifier to the training data (X_train, y_train)
rf_w.fit(X_train, y_train)

# Predict the labels for the test data (X_test)
y_pred = rf_w.predict(X_test)
import matplotlib.pyplot as plt
import numpy
from sklearn import metrics

# Compute the confusion matrix for the predictions
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

# Create a ConfusionMatrixDisplay object to visualize the confusion matrix
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])

# Plot the confusion matrix
cm_display.plot()
plt.show()

# Create a RandomForestClassifier with class weights {0: 0.05, 1: 0.95}
rf_w = RandomForestClassifier(class_weight={0: 0.05, 1: 0.95})

# Fit the RandomForestClassifier to the training data (X_train, y_train)
rf_w.fit(X_train, y_train)

# Predict the labels for the test data (X_test)
y_pred = rf_w.predict(X_test)

# Evaluate the model and print the classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Plot the confusion matrix using plot_confusion_matrix from mlxtend.plotting
from mlxtend.plotting import plot_confusion_matrix
plot_confusion_matrix(confusion_matrix(y_test, y_pred))

# Create a RandomForestClassifier with class_balance=True, which automatically balances class weights
rf_w = RandomForestClassifier(class_balance=True)

# Fit the RandomForestClassifier to the training data (X_train, y_train)
rf_w.fit(X_train, y_train)

# Predict the labels for the test data (X_test)
y_pred = rf_w.predict(X_test)

# Evaluate the model and print the classification report
print(classification_report(y_test, y_pred))

# Plot the confusion matrix using plot_confusion_matrix from mlxtend.plotting
plot_confusion_matrix(confusion_matrix(y_test, y_pred))

# Plot the ROC curve using RocCurveDisplay
from sklearn.metrics import RocCurveDisplay
ax = plt.gca()
rfc_disp = RocCurveDisplay.from_estimator(rf_w, X_test, y_test, ax=ax, alpha=0.8)
plt.show()


# Get feature importance from the previously fitted RandomForestClassifier (rf)
importance = rf.feature_importances_

# Summarize feature importance
for i, v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i, v))

# Plot feature importance
import matplotlib.pyplot as plt
plt.bar([x for x in range(len(importance))], importance)
plt.show()

# Calculate the correlation matrix for df_morgan_filtervar
corr_matrix_morgan_filtervar = df_morgan_filtervar.corr()

# Plot heatmap of the correlation matrix
plt.figure(figsize=(20, 15))
sns.heatmap(corr_matrix_morgan_filtervar, cmap="Greens")
plt.show()

# Define a function to find highly correlated features
def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

# Find highly correlated features for df_normalized_f_filtervar with a threshold of 0.9
corr_features_nf = correlation(df_normalized_f_filtervar, 0.9)
print(len(set(corr_features_nf)))
print(corr_features_nf)

# Find highly correlated features for df_morgan_filtervar with a threshold of 0.9
corr_features_morgan = correlation(df_morgan_filtervar, 0.9)
print(len(set(corr_features_morgan)))
print(corr_features_morgan)

# Prepare input data using OrdinalEncoder
def prepare_inputs(X_train, X_test):
    oe = OrdinalEncoder()
    oe.fit(X_train)
    X_train_enc = oe.transform(X_train)
    X_test_enc = oe.transform(X_test)
    return X_train_enc, X_test_enc

# Prepare target data using LabelEncoder
def prepare_targets(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc

# Assign features and labels
X = df_morgan_filtervar
y = label

# Split dataset into training set and test set (you can uncomment the lines below if needed)
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Call the functions to prepare input and target data
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# prepare input data
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
# prepare output data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)

def select_features(X_train, y_train, X_test, k_value='all'):
    fs = SelectKBest(score_func=chi2, k=k_value)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train_enc, y_train_enc, X_test_enc)
# what are scores for the features
for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))

# what are scores for the features
names = []
values = []
for i in range(len(fs.scores_)):
    names.append(X.columns[i])
    values.append(fs.scores_[i])
chi_list = zip(names, values)

# plot the scores
plt.figure(figsize=(10,4))
sns.barplot(x=names, y=values)
plt.xticks(rotation = 90)
plt.show()

# Import necessary libraries
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
import numpy as np

# Standardize the features in 'df_normalized_f_filtervar'
standard = preprocessing.scale(df_normalized_f_filtervar)
print(standard)

# Split the data into training and test sets
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(df_normalized_f_filtervar, label, test_size=0.25, random_state=1)

# Standardize the dataset; Standardization is essential before applying PCA
sc = StandardScaler()
sc.fit(X_train_n)
X_train_std = sc.transform(X_train_n)
X_test_std = sc.transform(X_test_n)

# Apply Principal Component Analysis (PCA) to determine transformed features
pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

# Print the count of different labels in the target variable 'y'
print(y.groupby('label').size())

# Visualize the distribution of target labels using a countplot
sns.countplot(x="label", data=y)

# Instantiate a Logistic Regression model with class_weight='balanced'
# 'balanced' automatically adjusts class weights based on the number of samples in each class.
lr = LogisticRegression(class_weight='balanced')

# Fit the Logistic Regression model on the training data
lr.fit(X_train, y_train)

# Predict the target labels for the test data using the trained model
y_pred = lr.predict(X_test)

# Evaluate the model's performance using classification_report and plot_confusion_matrix
print(classification_report(y_test, y_pred))
plot_confusion_matrix(confusion_matrix(y_test, y_pred))

# Define a range of class weight parameters for GridSearchCV
weights = np.linspace(0.05, 0.95, 20)

# Perform GridSearchCV to find the best class weight parameter for Logistic Regression
gsc = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid={
        'class_weight': [{0: x, 1: 1.0-x} for x in weights]
    },
    scoring='f1',
    cv=3
)
grid_result = gsc.fit(X, y)

# Print the best parameters found by GridSearchCV
print("Best parameters : %s" % grid_result.best_params_)

# Plot the weights vs f1 score
dataz = pd.DataFrame({ 'score': grid_result.cv_results_['mean_test_score'],
                       'weight': weights })
dataz.plot(x='weight')


# Do Grid Search again

weights = np.linspace(0.01, 0.05, 20)

gsc = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid={
        'class_weight': [{0: x, 1: 1.0-x} for x in weights]
    },
    scoring='f1',
    cv=3
)
grid_result = gsc.fit(X, np.ravel(y))

print("Best parameters : %s" % grid_result.best_params_)

# Plot the weights vs f1 score
dataz_2 = pd.DataFrame({ 'score': grid_result.cv_results_['mean_test_score'],
                       'weight': weights })
dataz_2.plot(x='weight')


# Best parameters : {'class_weight': {0: 0.04789473684210526, 1: 0.9521052631578948}}

# Import necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

# Instantiate a Logistic Regression model with specified class weights
lr = LogisticRegression(class_weight={0: 0.05, 1: 0.95})

# Fit the Logistic Regression model on the training data
lr.fit(X_train, y_train)

# Predict the target labels for the test data using the trained model
y_pred = lr.predict(X_test)

# Calculate the confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

# Display the confusion matrix using ConfusionMatrixDisplay
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
cm_display.plot()
plt.show()

# Instantiate an SVM classifier with specified class weights
clf = svm.SVC(kernel='linear', class_weight={0: 0.05, 1: 0.95})

# Fit the SVM classifier on the training data
clf = clf.fit(X_train, y_train)

# Predict the target labels for the test data using the trained model
y_pred = clf.predict(X_test)

# Calculate the confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

# Display the confusion matrix using ConfusionMatrixDisplay
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
cm_display.plot()
plt.show()

# Import necessary libraries
from sklearn.metrics import roc_curve, auc

# Run SVM classifier with linear kernel and enable probability estimates
classifier = svm.SVC(kernel='linear', probability=True)

# Fit the training data to the classifier and predict probabilities for the test data
probas_ = classifier.fit(X_train, y_train).predict_proba(X_test)

# Compute the ROC curve and area under the curve (AUC) for the SVM with linear kernel
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

# Compute the ROC curve and AUC for the SVM with RBF kernel (using the previously fitted SVM classifier)
test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_pred)

# Plot the ROC curve and AUC for the SVM with RBF kernel
plt.grid()
plt.plot(test_fpr, test_tpr, label="AUC TEST =" + str(auc(test_fpr, test_tpr)))
plt.plot([0, 1], 'g--')  # Diagonal line
plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUC (ROC curve)")
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.show()

# SVM classifier with RBF kernel and class weights {0: 0.05, 1: 0.95}
clf = svm.SVC(kernel='rbf', class_weight={0: 0.05, 1: 0.95})

# Fit the training data to the SVM classifier
clf.fit(X_train, y_train)

# Predict target labels for the test data using the trained model
y_pred = clf.predict(X_test)

# Calculate the confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

# Display the confusion matrix using ConfusionMatrixDisplay
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
cm_display.plot()
plt.show()

# Print the classification report showing precision, recall, F1-score, and support
print(classification_report(y_test, y_pred))

# Import necessary libraries
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Read the data from CSV files
train_data = pd.read_csv('train_full.csv', index_col=0)
df_new = pd.read_csv('features_new.csv', index_col=0)

# Separate features and labels
X = df_new[df_new.index.isin(train_data.index)].iloc[:, 0:-1]
y = train_data

# Standardize the features using StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
train_X = sc_X.fit_transform(X)
train_y = sc_y.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.25, random_state=1)

# Initialize SVM classifier with RBF kernel and class weights {0: 0.05, 1: 0.95}
clf = SVC(kernel='rbf', class_weight={0: 0.05, 1: 0.95})

# Fit the training data to the SVM classifier
clf.fit(X_train, y_train)

# Predict target labels for the test data using the trained SVM model
y_pred = clf.predict(X_test)

# Calculate the confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)

# Display the confusion matrix using ConfusionMatrixDisplay
cm_display = plot_confusion_matrix(clf, X_test, y_test, display_labels=[False, True])
cm_display.plot()
plt.show()

# Print the classification report showing precision, recall, F1-score, and support
print(classification_report(y_test, y_pred))

# Create a pipeline with SMOTE and Logistic Regression
pipe = make_pipeline(
    SMOTE(sampling_strategy=0.45),
    LogisticRegression()
)

# Fit the pipeline on the training data
pipe.fit(X_train, y_train)

# Predict target labels for the test data using the fitted pipeline
y_pred = pipe.predict(X_test)

# Print the classification report showing precision, recall, F1-score, and support for the pipeline
print(classification_report(y_test, y_pred))

# Perform Grid Search to find the best sampling_strategy for SMOTE
weights = np.linspace(0.005, 0.05, 10)
gsc = GridSearchCV(
    estimator=pipe,
    param_grid={
        'smote__sampling_strategy': weights
    },
    scoring='f1',
    cv=3
)
grid_result = gsc.fit(X_train, y_train)

print("Best parameters : %s" % grid_result.best_params_)

# Plot the weights vs f1 score for SMOTE Grid Search
dataz = pd.DataFrame({ 'score': grid_result.cv_results_['mean_test_score'],
                       'weight': weights })
dataz.plot(x='weight')
plt.show()

# Perform Grid Search to find the best class_weight for Logistic Regression
weights = np.linspace(0.05, 0.5, 10)
gsc = GridSearchCV(
    estimator=pipe,
    param_grid={
        'smote__sampling_strategy': weights
    },
    scoring='f1',
    cv=3
)
grid_result = gsc.fit(X_train, y_train)

print("Best parameters : %s" % grid_result.best_params_)

# Plot the weights vs f1 score for class_weight Grid Search
dataz = pd.DataFrame({ 'score': grid_result.cv_results_['mean_test_score'],
                       'weight': weights })
dataz.plot(x='weight')
plt.show()

# Train a RandomForestClassifier with different class weights
weights = np.linspace(0.01, 0.99, 20)
gsc = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid={
        'class_weight': [{0: x, 1: 1.0-x} for x in weights]
    },
    scoring='f1',
    cv=3
)
grid_result = gsc.fit(X, np.ravel(y))

print("Best parameters : %s" % grid_result.best_params_)

# Plot the weights vs f1 score for RandomForestClassifier Grid Search
dataz_2 = pd.DataFrame({ 'score': grid_result.cv_results_['mean_test_score'],
                       'weight': weights })
dataz_2.plot(x='weight')
plt.show()

# Fit RandomForestClassifier with best class_weight
rf_w = RandomForestClassifier(**grid_result.best_params_)
rf_w.fit(X_train, y_train)

# Predict target labels for the test data using the fitted RandomForestClassifier
y_pred = rf_w.predict(X_test)

# Print the classification report showing precision, recall, F1-score, and support for the RandomForestClassifier
print(classification_report(y_test, y_pred))
