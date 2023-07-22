#!/usr/bin/env python
# coding: utf-8

# Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

# Read data from CSV files
df_a = pd.read_csv('data_smile.csv')
df_b = pd.read_csv('oversampling_data.csv')

# Merge the two dataframes based on the 'smiles' column using right join
merged_df = pd.merge(df_a, df_b, on='smiles', how='right')
print(merged_df)

# Calculate confusion matrix using the merged data
confusion_matrix = metrics.confusion_matrix(merged_df['label_x'], round(merged_df['label_y']))

# Display the confusion matrix using ConfusionMatrixDisplay
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])

# Plot the confusion matrix
cm_display.plot()
plt.show()

# Creating a dataframe with performance data for different models
data = {
    'Model': ['Logistic Regression', 'SVM', 'RF', 'Chemprop', 'Oversampling Chemprop', 'Undersampling Chemprop (100)', 'Undersampling Chemprop (1000)'],
    'Random split': [0.442, 0.504, 0.202, 0, 0.968, 0.918, 0.908]
}

# Create a dataframe from the data
df = pd.DataFrame(data)

# Plotting a grouped bar chart to show the performance of classification models
ax = df.plot(x='Model', kind='bar', color='green', title='Performance of classification models', alpha=0.5)
ax.set_xticklabels(df['Model'], rotation=30, fontsize=7)
plt.legend(fontsize=8)
plt.ylabel('Recall')
plt.show()
