#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import chemprop
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.mixture import GaussianMixture

# Function to plot parity between true and predicted values
def plot_parity(y_true, y_pred, y_pred_unc=None):
    # Determine axis limits and calculate MAE and RMSE
    axmin = min(min(y_true), min(y_pred)) - 0.1*(max(y_true)-min(y_true))
    axmax = max(max(y_true), max(y_pred)) + 0.1*(max(y_true)-min(y_true))
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    
    # Plot the parity plot
    plt.plot([axmin, axmax], [axmin, axmax], '--k')
    plt.errorbar(y_true, y_pred, yerr=y_pred_unc, linewidth=0, marker='o', markeredgecolor='w', alpha=1, elinewidth=1)
    plt.xlim((axmin, axmax))
    plt.ylim((axmin, axmax))
    
    ax = plt.gca()
    ax.set_aspect('equal')
    
    # Add MAE and RMSE information as an anchored text
    at = AnchoredText(f"MAE = {mae:.2f}\nRMSE = {rmse:.2f}", prop=dict(size=10), frameon=True, loc='upper left')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    
    plt.xlabel('True')
    plt.ylabel('Chemprop Predicted')
    
    plt.show()
    
# Read preprocessed data from CSV files
df_normalized_f_filtervar = pd.read_csv('normalized_filtervar.csv')
label = pd.read_csv('label.csv', index_col=0)
df_normalized_f_filtervar.reset_index(drop=True, inplace=True)
label.reset_index(drop=True, inplace=True)
data_smile = pd.concat([df_normalized_f_filtervar['smiles'], label], axis=1)

# Save the smiles and labels data to a new CSV file
data_smile = data_smile.astype({'smiles': 'string'})
data_smile.to_csv('data_smile.csv')

# Function to convert SMILES to numerical feature vectors using Morgan fingerprints
def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    features = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)  # Using Morgan fingerprints as features
    features = np.array(features, dtype=float)
    return features

# Generate a set of SMILES structures for simulation
smiles_data = data_smile['smiles'].tolist()

# Convert SMILES to numerical feature vectors
X = np.array([smiles_to_features(smiles) for smiles in smiles_data])

# Use PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Use Gaussian Mixture Model (GMM) for modeling
gmm = GaussianMixture(n_components=5)
gmm.fit(X_pca)

# Sample from the GMM model to generate new feature vectors
n_samples = 107366  # Number of generated samples
generated_samples = gmm.sample(n_samples)
generated_features = generated_samples[0]

# Transform the generated feature vectors back to the original scale
generated_features_pca = pca.inverse_transform(generated_features)

# Output the generated SMILES structures
generated_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles_data[0])) for _ in range(n_samples)]
print("Generated SMILES:")
print(generated_smiles)

# Merge the original SMILES with the generated SMILES and assign a label of 1 to all data points
merged_list = smiles_data.copy()
merged_list.extend(generated_smiles)
data = {'smiles': merged_list, 'label': [1]*len(merged_list)}
df = pd.DataFrame(data)

# Save the data to a new CSV file named 'GTM.csv'
df.to_csv('GTM.csv')


# Import additional libraries
from sklearn import metrics

# Read data from CSV files
df_normalized_f_filtervar = pd.read_csv('normalized_filtervar.csv')
label = pd.read_csv('label.csv', index_col=0)
df_normalized_f_filtervar.reset_index(drop=True, inplace=True)
label.reset_index(drop=True, inplace=True)
data_smile = pd.concat([df_normalized_f_filtervar['smiles'], label], axis=1)

# Data preparation and analysis
data_T = data_smile[data_smile['label'] == 1]
data_F = data_smile[data_smile['label'] == 0]
data_smile_1 = data_smile
data_smile_1['pos'] = data_smile_1['label']
data_smile_1['one'] = np.full([108528, 1], 1)
data_smile_1['neg'] = data_smile_1['one'] - data_smile_1['pos']
data_smile_1.to_csv("data_smile_1.csv")

# Classification using chemprop library
arguments = [
    '--data_path', 'data_smile.csv',
    '--dataset_type', 'classification',
    '--save_dir', 'test_checkpoints',
     '--epochs', '5',
    '--save_smiles_splits',
    '--number_of_molecules', '1',
    '--smiles_column', 'smiles',
    '--target_columns','label',
    '--features_generator','morgan'
   
]

args = chemprop.args.TrainArgs().parse_args(arguments)
mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)

# Classification using chemprop with dropout
arguments = [
    '--data_path', 'data_smile.csv',
    '--dataset_type', 'classification',
    '--save_dir', 'test_checkpoints',
     '--epochs', '5',
    '--save_smiles_splits',
    '--number_of_molecules', '1',
    '--smiles_column', 'smiles',
    '--target_columns','label',
    '--class_balance',
    '--features_generator','morgan',
    '--depth','5',
    '--dropout','0.1', 
    '--ffn_num_layers','1', 
    '--hidden_size','1700' 
]

args = chemprop.args.TrainArgs().parse_args(arguments)
mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)

# Classification using chemprop with class weight
arguments = [
    '--data_path', 'data_smile_1.csv',
    '--dataset_type', 'classification',
    '--save_dir', 'test_checkpoints',
     '--epochs', '5',
    '--save_smiles_splits',
    '--number_of_molecules', '1',
    '--smiles_column', 'smiles',
    '--target_columns','pos','neg',
    '--target_weights','0.95','0.05'
]

args = chemprop.args.TrainArgs().parse_args(arguments)
mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)

# Classification using chemprop with dropout and target weight
arguments = [
    '--data_path', 'data_smile_1.csv',
    '--dataset_type', 'classification',
    '--save_dir', 'test_checkpoints',
     '--epochs', '5',
    '--save_smiles_splits',
    '--number_of_molecules', '1',
    '--smiles_column', 'smiles',
    '--target_columns','pos','neg',
    '--target_weights','0.95','0.05',
    '--features_generator','morgan',
    '--depth','5',
    '--dropout','0.1', 
    '--ffn_num_layers','1', 
    '--hidden_size','1700' 
]

args = chemprop.args.TrainArgs().parse_args(arguments)
mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)

# Classification using random forest
arguments_rf = [
    '--data_path', 'data_smile.csv',
    '--dataset_type', 'classification',
    '--model_type','random_forest',
    '--smiles_column', 'smiles',
    '--target_columns','label'
]

args_rf = chemprop.args.SklearnTrainArgs().parse_args(arguments_rf)
mean_score, std_score = chemprop.train.cross_validate(args=args_rf, train_func=chemprop.sklearn_train.run_sklearn)

# Classification using svm
arguments_rf = [
    '--data_path', 'data_smile.csv',
    '--dataset_type', 'classification',
    '--model_type','svm',
    '--smiles_column', 'smiles',
    '--target_columns','label'
]

args_rf = chemprop.args.SklearnTrainArgs().parse_args(arguments_rf)
mean_score, std_score = chemprop.train.cross_validate(args=args_rf, train_func=chemprop.sklearn_train.run_sklearn)

# Classification using svm with balanced classes
arguments_rf = [
    '--data_path', 'data_smile.csv',
    '--dataset_type', 'classification',
    '--model_type','svm',
    '--smiles_column', 'smiles',
    '--target_columns','label',
    '--class_weight','balanced'
]

args_rf = chemprop.args.SklearnTrainArgs().parse_args(arguments_rf)
mean_score, std_score = chemprop.train.cross_validate(args=args_rf, train_func=chemprop.sklearn_train.run_sklearn)

# Regression with balanced classes
arguments_rf = [
    '--data_path', 'data_score.csv',
    '--dataset_type', 'regression',
    '--model_type','random_forest',
    '--smiles_column', 'smiles',
    '--target_columns','enrichment_score',
    '--class_weight','balanced',
    '--save_dir', 'test_checkpoints_rf',
    '--save_smiles_splits'
]

args_rf = chemprop.args.SklearnTrainArgs().parse_args(arguments_rf)
mean_score, std_score = chemprop.train.cross_validate(args=args_rf, train_func=chemprop.sklearn_train.run_sklearn)


arguments = [
    '--test_path', 'test_checkpoints_rf/fold_0/test_smiles.csv',
    '--checkpoint_dir', 'test_checkpoints_rf',#Directory where the model checkpoint(s) are saved (i.e. --save_dir during training).
    '--preds_path', 'test_preds_randomforest.csv',
    '--smiles_column', 'smiles'
]

args = chemprop.args.SklearnPredictArgs().parse_args(arguments)
preds = chemprop.sklearn_predict.predict_sklearn(args=args)


# using this model to predict
preds_randomforest=pd.read_csv('test_preds_randomforest.csv')
df_rf = pd.read_csv('test_checkpoints_rf/fold_0/test_full.csv')
df_rf['preds'] = preds_randomforest['enrichment_score']

plot_parity(df_rf.enrichment_score, df_rf.preds)

# Regression with morgan fingerprints
arguments_p = [
    '--test_path', 'prediction.csv',
    '--checkpoint_dir', 'test_checkpoints',
# Directory where the model checkpoint(s) are saved (i.e. --save_dir during training).
    '--preds_path', 'test_preds.csv',
    '--smiles_column', 'smiles',
    #'--target_columns','pos','neg',
    #'--target_weights','0.95','0.05',
    '--features_generator','morgan'
]

args_p = chemprop.args.PredictArgs().parse_args(arguments_p)
preds = chemprop.train.make_predictions(args=args_p)


# Regression with classification uncertainty
arguments_p_all = [
    #'--test_path', 'prediction_all.csv',
    '--test_path', 'test_checkpoints_base/fold_0/test_smiles.csv',
    '--checkpoint_dir', 'test_checkpoints_base',#Directory where the model checkpoint(s) are saved (i.e. --save_dir during training).
    '--preds_path', 'test_preds_all.csv',
    '--smiles_column', 'smiles',
    '--uncertainty_method','classification'
]

args_p_all = chemprop.args.PredictArgs().parse_args(arguments_p_all)
preds = chemprop.train.make_predictions(args=args_p_all)




df = pd.read_csv('test_checkpoints_base/fold_0/test_full.csv')
df['preds'] = [x[0] for x in preds]

plot_parity(df.label, df.preds)

# Regression baseline
arguments_p_all = [
    '--test_path', 'test_checkpoints_base_regression/fold_0/test_smiles.csv',
    '--checkpoint_dir', 'test_checkpoints_base_regression',#Directory where the model checkpoint(s) are saved (i.e. --save_dir during training).
    '--preds_path', 'test_preds_base_regression.csv',
    '--smiles_column', 'smiles'
]

args_p_all = chemprop.args.PredictArgs().parse_args(arguments_p_all)
preds = chemprop.train.make_predictions(args=args_p_all)



df = pd.read_csv('test_checkpoints_base_regression/fold_0/test_full.csv')
df['preds'] = [x[0] for x in preds]

plot_parity(df.enrichment_score, df.preds)


import numpy as np
import matplotlib.pyplot as plt


# creating the dataset
data = {'Randomforest':0.743, 'SVM':0.875, 'Chemprop':0.910}
courses = list(data.keys())
values = list(data.values())

fig = plt.figure(figsize = (6, 5))

# creating the bar plot
plt.bar(courses, values,width = 0.4)
plt.xlabel("Model")
plt.ylabel("AUC")
plt.title("Model Performance")
plt.show()

# importing package
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# create data
df = pd.DataFrame([['RF', 1.345371, 1.296126], ['SVM', 1.261808 ,1.242779], ['Chemprop', 1.217367, 1.205343]],
                  columns=['Model', 'Random split', 'Scaffold split'])
# view data
df

# plot grouped bar chart
df.plot(x='Model',
        kind='bar',
        rot=0,
        stacked=False,
        fontsize='large',
        title='Performance of regression model',
        alpha=0.5)
plt.ylim(1,1.4)
plt.legend(fontsize = 8)
plt.ylabel('RMSE')

#1. Get the number of positive (fraud) samples in the data
pos = len(data_smile[data_smile['label'] == 1])
#2. Get indices of non fraud samples
neg_indices = data_smile[data_smile.label == 0].index
#3. Random sample non fraud indices
random_indices = np.random.choice(neg_indices,pos, replace=False)
#4. Find the indices of fraud samples
neg_indices = data_smile[data_smile.label == 1].index
#5. Concat fraud indices with sample non-fraud ones
under_sample_indices = np.concatenate([neg_indices,random_indices])
#6. Get Balance Dataframe
under_sample = data_smile.loc[under_sample_indices]

under_sample.to_csv('under_sample.csv')
under_sample


for i in range(100):
#1. Get the number of positive (fraud) samples in the data
    pos = len(data_smile[data_smile['label'] == 1])
#2. Get indices of non fraud samples
    neg_indices = data_smile[data_smile.label == 0].index
#3. Random sample non fraud indices
    np.random.seed(i)
    random_indices = np.random.choice(neg_indices, pos, replace=False)
#4. Find the indices of fraud samples
    neg_indices = data_smile[data_smile.label == 1].index
#5. Concat fraud indices with sample non-fraud ones
    under_sample_indices = np.concatenate([neg_indices,random_indices])
#6. Get Balance Dataframe
    under_sample = data_smile.loc[under_sample_indices]
    under_sample.to_csv('under_sample_{}.csv'.format(i))

# interate undersampling classification chemprop for 100 times
for i in range(0,100,1):
    pathway = 'under_sample_{}.csv'.format(i)
    saveway= 'test_checkpoints_under_{}'.format(i)
    arguments = [
    '--data_path',pathway,
    '--dataset_type', 'classification',
    '--save_dir', saveway,
    '--epochs', '5',
    '--save_smiles_splits',
    '--number_of_molecules', '1',
    '--smiles_column', 'smiles',
    '--target_columns','label',
    '--features_generator','morgan'
]
    args = chemprop.args.TrainArgs().parse_args(arguments)
    mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)

# make predection for 100 times
for i in range(0,100,1):
    testway = 'test_checkpoints_under_{}/fold_0/test_smiles.csv'.format(i)
    checkway='test_checkpoints_under_{}'.format(i)
    predway='test_checkpoints_under_{}.csv'.format(i)
    arguments = [
    '--test_path', testway,
    '--checkpoint_dir', checkway,#Directory where the model checkpoint(s) are saved (i.e. --save_dir during training).
    '--preds_path', predway,
    '--smiles_column', 'smiles',
    '--features_generator','morgan'
]

    args = chemprop.args.PredictArgs().parse_args(arguments)
    preds = chemprop.train.make_predictions(args=args)

import os
import glob
import pandas as pd
#os.chdir("/folder_test_under")

# save the 100 files
extension = 'csv'
all_filenames = [i for i in glob.glob('test_checkpoints_under_*[0-99]-Copy1.*'.format(extension))]
all_filenames


# Combine all files in the list 'all_filenames' into a single DataFrame 'combined_csv'
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])

# Export the combined DataFrame to a CSV file named "combined_csv.csv"
combined_csv.to_csv("combined_csv.csv", index=False, encoding='utf-8-sig')

# Round the 'label' column in the 'combined_csv' DataFrame
combined_csv['label'] = round(combined_csv['label'])

# Group the data in 'combined_csv' by 'smiles' and calculate the mean for each group
df_test = combined_csv.groupby(['smiles']).agg('mean')
df_test.index.name = 'smiles'
df_test.reset_index(inplace=True)

# Merge the grouped data with the original 'data_smile' DataFrame on the 'smiles' column
merged_df = pd.merge(df_test, data_smile, on='smiles')

# Round the 'label_x' column (mean label) to create a new column 'label_round'
merged_df['label_round'] = round(merged_df['label_x'])

# Display the merged DataFrame
print(merged_df)

# Count the number of positive (fraud) samples in the 'label_y' column of the merged DataFrame
num_positive_samples = (merged_df['label_y'] == 1).sum()

# Print the number of positive samples
print(num_positive_samples)

# Calculate the confusion matrix using the 'label_y' and 'label_round' columns of the merged DataFrame
confusion_matrix = metrics.confusion_matrix(merged_df['label_y'], merged_df['label_round'])

# Create a ConfusionMatrixDisplay object with the confusion matrix and display labels
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])

# Plot the confusion matrix
cm_display.plot()
plt.show()

# Create a cross-tabulation (contingency table) of 'label_y' and 'label_round' columns in the merged DataFrame
cross_tab = pd.crosstab(merged_df['label_y'], merged_df['label_round'], rownames=['Actual'], colnames=['Predicted'])

# Print the cross-tabulation
print(cross_tab)

# Calculate the AUC (Area Under the Curve) of the model using the 'label_y' and 'label_round' columns in the merged DataFrame
auc = metrics.roc_auc_score(merged_df['label_y'], merged_df['label_round'])

# Print the AUC score
print(auc)

# Plot a parity plot to compare the true labels ('label_y') with the predicted labels ('label_x') in the merged DataFrame
plot_parity(merged_df['label_y'], merged_df['label_x'])


#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
#export to csv
combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')

combined_csv['label']=round(combined_csv['label'])

# Group the data in 'combined_csv' by 'smiles' and calculate the median for each group
df_test=combined_csv.groupby(['smiles']).agg('median')
df_test.index.name = 'smiles'
df_test.reset_index(inplace=True)
# Create a cross-tabulation (contingency table) of 'label_y' and 'label_round' columns in the merged DataFrame
merged_df = pd.merge(df_test,data_smile,on='smiles')
merged_df
pd.crosstab(merged_df['label_y'], merged_df['label_x'], rownames=['Actual'], colnames=['Predicted'])


y_test = merged_df.loc[:, merged_df.columns == 'label_x']
y_pred = merged_df.loc[:, merged_df.columns == 'label_y']

confusion_matrix = pd.crosstab(merged_df['label_y'], merged_df['label_x'], rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)

sns.heatmap(pd.crosstab(merged_df['label_y'], merged_df['label_x'], rownames=['Actual'], colnames=['Predicted']),
            cmap="YlGnBu", annot=True, cbar=False, fmt='g')

#1. Iterate 1000 times to ensemble
for i in range(1000):
    pos = len(data_smile[data_smile['label'] == 1])
#2. Get indices of non fraud samples
    neg_indices = data_smile[data_smile.label == 0].index
#3. Random sample non fraud indices
    np.random.seed(i)
    random_indices = np.random.choice(neg_indices, pos, replace=False)
    #neg_indices-random_indices
#4. Find the indices of fraud samples
    neg_indices = data_smile[data_smile.label == 1].index
#5. Concat fraud indices with sample non-fraud ones
    under_sample_indices = np.concatenate([neg_indices,random_indices])
#6. Get Balance Dataframe
    under_sample = data_smile.loc[under_sample_indices]
    under_sample.to_csv('chemprop_undersampling/under_sample_{}.csv'.format(i))

random_indices.delete(neg_indices)

neg_indices

# Ensemble the 
for i in range(100,1000,1):
    pathway = 'chemprop_undersampling/under_sample_{}.csv'.format(i)
    saveway= 'chemprop_undersampling/test_checkpoints_under_{}'.format(i)
    arguments = [
    '--data_path',pathway,
    '--dataset_type', 'classification',
    '--save_dir', saveway,
    '--epochs', '5',
    '--save_smiles_splits',
    '--number_of_molecules', '1',
    '--smiles_column', 'smiles',
    '--target_columns','label',
    '--features_generator','morgan'
]
    args = chemprop.args.TrainArgs().parse_args(arguments)
    mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)


for i in range(657,1000,1):
    testway = 'chemprop_undersampling/test_checkpoints_under_{}/fold_0/test_smiles.csv'.format(i)
    checkway='chemprop_undersampling/test_checkpoints_under_{}'.format(i)
    predway='chemprop_undersampling/test_checkpoints_under_{}.csv'.format(i)
    arguments = [
    '--test_path', testway,
    '--checkpoint_dir', checkway,#Directory where the model checkpoint(s) are saved (i.e. --save_dir during training).
    '--preds_path', predway,
    '--smiles_column', 'smiles',
    '--features_generator','morgan'
]

    args = chemprop.args.PredictArgs().parse_args(arguments)
    preds = chemprop.train.make_predictions(args=args)

# Save the files
extension = 'csv'
all_filenames = [i for i in glob.glob('chemprop_undersampling/test_checkpoints_under_*[0-999].*'.format(extension))]

#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
#export to csv
combined_csv.to_csv( "chemprop_undersampling/combined_csv.csv", index=False, encoding='utf-8-sig')

# Ensemble the undersampling model by mean
df_test=combined_csv.groupby(['smiles']).agg('mean')
df_test.index.name = 'smiles'
df_test.reset_index(inplace=True)

merged_df = pd.merge(df_test,data_smile,on='smiles')
merged_df['label_round']=round(merged_df['label_x'])
merged_df

#combined_csv

(merged_df['label_y'] == 1).sum()

# make confusion matrix of the ensemble data
confusion_matrix = metrics.confusion_matrix(merged_df['label_y'], merged_df['label_round'])

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()

#calculate AUC of model
auc = metrics.roc_auc_score(merged_df['label_y'], merged_df['label_round'])

#print AUC score
print(auc)


from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
# calculate roc curve
fpr, tpr, thresholds = roc_curve(merged_df['label_y'], merged_df['label_round'])

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
#export to csv
combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')

combined_csv['label']=round(combined_csv['label'])

# Ensemble the undersampling model by median
df_test=combined_csv.groupby(['smiles']).agg('median')
df_test.index.name = 'smiles'
df_test.reset_index(inplace=True)
# Merge df_test and data_smile DataFrames on the 'smiles' column to create merged_df
merged_df = pd.merge(df_test, data_smile, on='smiles')
merged_df

# Calculate the confusion matrix using the 'label_y' and 'label_x' columns of the merged DataFrame
confusion_matrix = pd.crosstab(merged_df['label_y'], merged_df['label_x'], rownames=['Actual'], colnames=['Predicted'])
confusion_matrix

# Import seaborn library for data visualization
import seaborn as sns

# Plot a heatmap of the confusion matrix using Seaborn
sns.heatmap(pd.crosstab(merged_df['label_y'], merged_df['label_x'], rownames=['Actual'], colnames=['Predicted']),
            cmap="YlGnBu", annot=True, cbar=False, fmt='g')

# Read the 'test_checkpoints_classweight_chemprop.csv' and 'test_full.csv' files
classweight = pd.read_csv('test_checkpoints_classweight_chemprop.csv')
classweight.index.name = 'smiles'

test = pd.read_csv('test_full.csv')
test.index.name = 'smiles'

# Merge the 'pos' columns of test and classweight DataFrames on the 'smiles' column to create merged_df
merged_df = pd.merge(test['pos'], classweight['pos'], on='smiles')
merged_df

# Round the values in the 'pos_y' column
merged_df['pos_y'].round

# Print the merged DataFrame
merged_df

# Calculate the confusion matrix using the 'pos_x' and rounded 'pos_y' columns of the merged DataFrame
confusion_matrix = metrics.confusion_matrix(merged_df['pos_x'], round(merged_df['pos_y']))

# Create a ConfusionMatrixDisplay object with the confusion matrix and display labels
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])

# Plot the confusion matrix
cm_display.plot()
plt.show()

import pandas as pd

# Read the 'test_checkpoints_scaffold_regression.csv' and 'data_score.csv' files
df_predict_chemprop_scaffold = pd.read_csv('test_checkpoints_scaffold_regression.csv')
df_predict_chemprop_scaffold = df_predict_chemprop_scaffold.set_index('smiles')

df_score = pd.read_csv('data_score.csv')
df_score = df_score.set_index('smiles')

# Filter and select the last column of df_score for the smiles present in df_predict_chemprop_scaffold
df_merge = df_score[df_score.index.isin(df_predict_chemprop_scaffold.index)].iloc[:, -1]
df_merge

# Merge the selected column from df_score and df_predict_chemprop_scaffold on the 'smiles' column
merged_df = pd.merge(df_merge, df_predict_chemprop_scaffold, on='smiles')
merged_df

import numpy as np
import matplotlib.pyplot as plt

# Define the x and y values for the scatter plot
x = merged_df.enrichment_score_x
y = merged_df.enrichment_score_y

def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # Remove labels from the histograms
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # Plot the scatter plot
    ax.scatter(x, y)

    # Determine the binwidth and limits for the histograms
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')

# Start with a square Figure.
fig = plt.figure(figsize=(6, 6))
# Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
# the size of the marginal axes and the main axes in both directions.
# Also adjust the subplot parameters for a square plot.
gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)
# Create the Axes.
ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

# Draw the scatter plot and marginals using the scatter_hist function
scatter_hist(x, y, ax, ax_histx, ax_histy)


# Function to create scatter plot with marginal histograms
def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # Remove labels from the histograms
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # Plot the scatter plot
    ax.scatter(x, y)

    # Determine the binwidth and limits for the histograms
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')


# Function to create a parity plot (scatter plot with 1:1 line)
def plot_parity(y_true, y_pred, y_pred_unc=None):
    # Calculate metrics: MAE and RMSE
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    
    # Define plot limits
    axmin = min(min(y_true), min(y_pred)) - 0.1*(max(y_true)-min(y_true))
    axmax = max(max(y_true), max(y_pred)) + 0.1*(max(y_true)-min(y_true))

    # Plot the 1:1 line
    plt.plot([axmin, axmax], [axmin, axmax], '--k')

    # Plot the data points
    plt.errorbar(y_true, y_pred, yerr=y_pred_unc, linewidth=0, marker='o', markeredgecolor='w', alpha=1, elinewidth=1)
    
    # Set plot limits
    plt.xlim((0, 10))
    plt.ylim((0, 10))
    
    ax = plt.gca()
    ax.set_aspect('equal')
    
    # Add anchored text with MAE and RMSE values
    at = AnchoredText(f"MAE = {mae:.2f}\nRMSE = {rmse:.2f}", prop=dict(size=10), frameon=True, loc='upper left')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    
    # Set axis labels
    plt.xlabel('True')
    plt.ylabel('Chemprop Predicted')
    
    # Show the plot
    plt.show()


# Create a scatter plot with point density
xy = np.vstack([merged_df.enrichment_score_x, merged_df.enrichment_score_y])
z = gaussian_kde(xy)(xy)

fig, ax = plt.subplots()
# Scatter plot with point density color-coded
ax.scatter(x, y, c=z, s=100)

# Set plot limits
plt.xlim([0, 5])
plt.ylim([0, 5])
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap

# "Viridis-like" colormap with white background
white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    #(1e-20, '#fde624'),
    (1e-20, '#ffffff'),
    (0.2, '#78d151'),
    (0.4, '#21a784'),
    (0.6, '#2a788e'),
    (0.8, '#404388'),
    (1, '#440053'),
], N=256)

def using_mpl_scatter_density(fig, x, y):
    # Add subplot with scatter_density projection
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    # Create the scatter density plot
    density = ax.scatter_density(x, y, cmap=white_viridis)
    # Add colorbar with label
    fig.colorbar(density, label='Number of points per pixel')

# Create a new figure
fig = plt.figure()
# Call the function to create the scatter density plot
using_mpl_scatter_density(fig, merged_df.enrichment_score_x, merged_df.enrichment_score_y)

# Set plot limits
plt.xlim([0, 5])
plt.ylim([0, 5])
# Show the plot
plt.show()

