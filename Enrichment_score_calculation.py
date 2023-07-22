#!/usr/bin/env python
# coding: utf-8

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotnine as p9

# RDKit is a cheminformatics library
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

# Setting up the environment for inline plotting
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns; sns.set()

# Reading the data from the CSV file
df = pd.read_csv('DD1S_CAIX_QSAR.csv')

# Calculating enrichment ratio
df.loc[:, 'enrichment_ratio'] = df['exp_tot'] / df['beads_tot']
enrich = df.loc[df['enrichment_ratio'] > 1]
df_hist = df.loc[df['beads_tot'] != 0]

# Grouping data by cycle1 and cycle2, calculating enrichment, and storing in group1_df
group1 = df.groupby(["cycle1", "cycle2"])
group1_exp = group1.sum()[['exp_tot']]
group1_beads = group1.sum()[['beads_tot']]
group1_df = pd.concat([group1_exp, group1_beads], axis=1, join='inner')
group1_df.loc[:, 'enrichment'] = group1_df['exp_tot'] / group1_df['beads_tot']

# Similarly, grouping data by cycle1 and cycle3, and calculating enrichment for group2_df
group2 = df.groupby(["cycle1", "cycle3"])
group2_exp = group2.sum()[['exp_tot']]
group2_beads = group2.sum()[['beads_tot']]
group2_df = pd.concat([group2_exp, group2_beads], axis=1, join='inner')
group2_df.loc[:, 'enrichment'] = group2_df['exp_tot'] / group2_df['beads_tot']

# Grouping data by cycle2 and cycle3, and calculating enrichment for group3_df
group3 = df.groupby(["cycle2", "cycle3"])
group3_exp = group3.sum()[['exp_tot']]
group3_beads = group3.sum()[['beads_tot']]
group3_df = pd.concat([group3_exp, group3_beads], axis=1, join='inner')
group3_df.loc[:, 'enrichment'] = group3_df['exp_tot'] / group3_df['beads_tot']

# Defining a function to calculate R from z values
def R_from_z(k2, n2, k1, n1, z):
    a = np.power(z, 2) / 4 - (k2 + 3/8)
    b = 2 * np.sqrt(k1 + 3/8) * np.sqrt(k2 + 3/8)
    c = np.power(z, 2) / 4 - (k1 + 3/8)
    x = (-b - np.sign(z) * np.sqrt(np.clip(np.power(b, 2) - 4 * a * c, 0, np.inf))) / (2 * a)
    return np.power(x, 2) * n2 / n1

# Calculating R values using the defined function and storing them in the DataFrame
k1 = df['exp_tot']
k2 = df['beads_tot']
n1 = 638831
n2 = 5208230
z = 0
df_new = pd.concat([df, R_from_z(k2, n2, k1, n1, z)], axis=1)
df_new.rename(columns={0: 'enrichment_score'}, inplace=True)

# Sorting the DataFrame based on enrichment_score and plotting the histogram
df_new_rank = df_new.sort_values(by="enrichment_score", axis=0, ascending=False)
df_new_rank.hist(column="enrichment_score", bins=200)

# Defining a function to label compounds as enriched or not based on enrichment_score
def func_enrich(df):
    if df["enrichment_score"] >= 8.152750884:
        return 1
    else:
        return 0

# Applying the function to the DataFrame and saving the result to a CSV file
df_new_rank.loc[:, "label"] = df_new_rank.apply(func_enrich, axis=1)
df_new_rank['label'].to_csv('label.csv')



import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans

# Describe the data in the DataFrame 'df_new_rank'
df_new_rank.describe()
# 1. Identify compounds where exp_tot is greater than beads_tot, indicating enrichment: >8.153 enriched
# 2. Identify compounds in the top quartile of the enrichment_ratio column: >1.469 enriched
# 3. Identify compounds that are 2 standard deviations above the mean in the enrichment_score column: >??

# TODO: Perform k-means clustering and different ways of tSNE and PCA on compounds using Morgan fingerprints as the starting features

# Enable SVG output for IPython
IPythonConsole.ipython_useSVG = True  #< set this to False if you want PNGs instead of SVGs

# Define a function to add atom indices to a molecule
def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

# Create a molecule from a SMILES string and add atom indices
mol = Chem.MolFromSmiles("[No]NC(=O)[C@@H]1[C@@H](c2ccc(-c3cccc4cnn(C)c34)cc2)CCN1C[C@H](O)[C@H]1OC(=O)[C@@H](O)[C@H]1O")
mol_with_atom_index(mol)

# Compute Gasteiger charges for the molecule and display them on atoms
AllChem.ComputeGasteigerCharges(mol)
mol2 = Chem.Mol(mol)
for at in mol2.GetAtoms():
    lbl = '%.2f' % (at.GetDoubleProp("_GasteigerCharge"))
    at.SetProp('atomNote', lbl)
mol2

# Create another molecule and find a substructure match
m = Chem.MolFromSmiles('[No]NC(=O)[C@H]1[C@@H](c2ccc(C=CC(C)(C)C)cc2)CN1Cc1ccc(CC)c([N+](=O)[O-])c1')
substructure = Chem.MolFromSmarts('[C@H]')
print(m.GetSubstructMatches(substructure))

# Import necessary libraries for data visualization and analysis
import pandas as pd
import tmap
from faerun import Faerun
from mhfp.encoder import MHFPEncoder
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools

# Select the relevant columns from the DataFrame 'df_new_rank'
df_new_rank_1 = df_new_rank[["smiles", "enrichment_score"]]
# Select the first 10 rows of the DataFrame
df_new_rank_2 = df_new_rank_1.head(10)
# Create a new column 'label' based on the enrichment_score value
df_new_rank_1['label'] = ["binded" if x > 8.15 else "not_binded" for x in df_new_rank_1['enrichment_score']]
# Add a molecule column to the DataFrame based on the SMILES column
PandasTools.AddMoleculeColumnToFrame(df_new_rank_1, smilesCol='smiles')

# Morgan Fingerprint (ECFPx)
# Set the radius and number of bits for the Morgan fingerprint
radius = 3 
nBits = 1024
# Compute the Morgan fingerprint for each molecule in the DataFrame
ECFP6 = [AllChem.GetMorganFingerprintAsBitVect(x, radius=radius, nBits=nBits) for x in df_new_rank_1['ROMol']]
# Create a DataFrame with the Morgan fingerprint as features
ecfp6_name = [f'Bit_{i}' for i in range(nBits)]
ecfp6_bits = [list(l) for l in ECFP6]
df_morgan = pd.DataFrame(ecfp6_bits, index=df_new_rank_1.smiles, columns=ecfp6_name)

# Read the Morgan fingerprint DataFrame from a CSV file
df_morgan = pd.read_csv('Fingerprint.csv', index_col=0)


# Compute the correlation matrix for the DataFrame 'df_morgan'
corr_matrix_morgan = df_morgan.corr()

# Plot a heatmap of the correlation matrix using Seaborn and Matplotlib
plt.figure(figsize=(20, 15))
sns.heatmap(corr_matrix_morgan, cmap="Greens")

# Perform variance thresholding to remove features with zero variance
var_threshold_2 = VarianceThreshold(threshold=0)
# Fit the data to the variance thresholding model
var_threshold_2.fit(df_morgan)

# Transform the data based on the variance thresholding model and print the transformed data
print(var_threshold_2.transform(df_morgan))
print('*' * 10, "Separator", '*' * 10)

# Print the shapes of the data before and after transformation
print("Earlier shape of data: ", df_morgan.shape)
print("Shape after transformation: ", var_threshold_2.transform(df_morgan).shape)

# Get the labels of features that passed the variance thresholding
var_label_morgan = var_threshold_2.get_support()
list_var_2 = [i for i, x in enumerate(var_label_morgan) if x]
# Filter the DataFrame 'df_morgan' based on the selected features
df_morgan_filtervar = df_morgan.iloc[:, list_var_2]

# Read the filtered DataFrame from a CSV file
df_morgan_filtervar = pd.read_csv('Fingerprint_filtervar.csv', index_col=0)

# Compute the correlation matrix for the filtered DataFrame 'df_morgan_filtervar'
corr_matrix_morgan_filtervar = df_morgan_filtervar.corr()

# Plot a heatmap of the correlation matrix for the filtered DataFrame using Seaborn and Matplotlib
plt.figure(figsize=(20, 15))
sns.heatmap(corr_matrix_morgan_filtervar, cmap="Greens")


# Define a function for correlation filtering
def corrFilter(x: pd.DataFrame, bound: float):
    xCorr = x.corr()
    xFiltered = xCorr[((xCorr >= bound) | (xCorr <= -bound)) & (xCorr != 1.000)]
    xFlattened = xFiltered.unstack().sort_values().drop_duplicates()
    return xFlattened

# Compute the number of highly correlated features with a correlation threshold of 0.9
len(corrFilter(corr_matrix_morgan_filtervar, .9))

# Print the DataFrame 'corr_matrix_nf_filtervar'
print(corr_matrix_nf_filtervar)


# Import the necessary libraries for descriptor generation
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator

# Create a generator for RDKit2D descriptors
generator = MakeGenerator(("RDKit2D",))

# Get the list of RDKit2D descriptor names
nf_list = []
for name, numpy_type in generator.GetColumns():
    nf_list.append(name)
    print("name: {} data type: {}".format(name, numpy_type))

# Exclude the first descriptor name as it is the molecule identifier
nf_list = nf_list[1:201]

# Import the generator for RDKit2D and Morgan3Counts descriptors
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
generator = MakeGenerator(("RDKit2D", "Morgan3Counts"))

# Create a list of SMILES strings from the 'smiles' column in the DataFrame 'df_new_rank_1'
smiles_0 = df_new_rank_1['smiles']

# Create a normalized descriptor generator for RDKit2D descriptors
from descriptastorus.descriptors import rdNormalizedDescriptors
from rdkit import Chem
import logging

# Create a generator for RDKit2DNormalized descriptors
generator = rdNormalizedDescriptors.RDKit2DNormalized()
generator.columns # list of tuples: (descriptor_name, numpytype) ...

# Function for converting a SMILES string into normalized RDKit2D descriptor values
def rdkit_2d_normalized_features(smiles: str):
    # n.b. the first element is true/false if the descriptors were properly computed
    results = generator.process(smiles)
    processed, features = results[0], results[1:]
    if processed is None:
       logging.warning("Unable to process smiles %s", smiles)
    # if processed is None, the features are default values for the type
    return features

# Compute the normalized RDKit2D descriptors for each SMILES string in the DataFrame 'smiles_0'
normalized_f = [rdkit_2d_normalized_features(x) for x in smiles_0]

# Create a DataFrame 'df_normalized_f' with the normalized RDKit2D descriptors as features
df_normalized_f = pd.DataFrame(normalized_f)
df_normalized_f.index = np.arange(1, len(df_normalized_f)+1)

# Save the DataFrame 'df_normalized_f' to an Excel file
resultExcelFile_nf = pd.ExcelWriter('normalized_features_final.xlsx')
df_normalized_f.to_excel(resultExcelFile_nf, index=False)
resultExcelFile_nf.save()

# Read the DataFrame 'df_normalized_f' from a CSV file and set the column names and indices
df_normalized_f = pd.read_csv('df_normalized_f_final.csv', index_col=0)
df_normalized_f.columns = nf_list
df_normalized_f.index = smiles_0

# Compute the correlation matrix for the DataFrame 'df_normalized_f'
corr_matrix_nf = df_normalized_f.corr()

# Plot a heatmap of the correlation matrix using Seaborn and Matplotlib
plt.figure(figsize=(20, 15))
sns.heatmap(corr_matrix_nf, cmap="Greens")

# Create a variance thresholding model
var_threshold = VarianceThreshold(threshold=0)

# Fit the data to the variance thresholding model
var_threshold.fit(df_normalized_f)

# Transform the data based on the variance thresholding model and print the transformed data
print(var_threshold.transform(df_normalized_f))
print('*' * 10, "Separator", '*' * 10)

# Print the shapes of the data before and after transformation
print("Earlier shape of data: ", df_normalized_f.shape)
print("Shape after transformation: ", var_threshold.transform(df_normalized_f).shape)

# Get the labels of features that passed the variance thresholding for normalized features
var_label_nf = var_threshold.get_support()
list_var_1 = [i for i, x in enumerate(var_label_nf) if x]
# Filter the DataFrame 'df_normalized_f' based on the selected features
df_normalized_f_filtervar = df_normalized_f.iloc[:, list_var_1]

# Read the filtered DataFrame from a CSV file
df_normalized_f_filtervar = pd.read_csv('normalized_filtervar.csv', index_col=0)

# Compute the correlation matrix for the filtered DataFrame 'df_normalized_f_filtervar'
corr_matrix_nf_filtervar = df_normalized_f_filtervar.corr()

# Plot a heatmap of the correlation matrix for the filtered DataFrame using Seaborn and Matplotlib
plt.figure(figsize=(20, 15))
sns.heatmap(corr_matrix_nf_filtervar, cmap="Greens")

# Calculate absolute correlation values for each pair of features in 'corr_matrix_nf_filtervar'
c1 = corr_matrix_nf_filtervar.abs().unstack()
# Sort the correlation values in descending order
c1_sorted = c1.sort_values(ascending=False)

# Transpose the correlation matrix and sort the values in descending order
table1 = corr_matrix_nf_filtervar[corr_matrix_nf_filtervar < 1].unstack().transpose()\
    .sort_values(ascending=False)\
    .drop_duplicates()

# Combine the filtered normalized features and Morgan fingerprints for the final feature set
X = pd.concat([df_normalized_f_filtervar[:], df_morgan_filtervar[:]], axis=1)  # Features
y = df_new_rank_1['label']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 

# Create a Logistic Regression classifier with balanced class weights
lr = LogisticRegression(class_weight='balanced')

# Fit the logistic regression model to the training data
lr.fit(X_train, y_train)

# Predict labels for the test set using the trained model
y_pred = lr.predict(X_test)

# Evaluate the model using classification report and plot the confusion matrix
print(classification_report(y_test, y_pred))
plot_confusion_matrix(confusion_matrix(y_test, y_pred))

# Perform a grid search to find the best class weights for Logistic Regression
weights = np.linspace(0.05, 0.95, 20)
gsc = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid={
        'class_weight': [{0: x, 1: 1.0-x} for x in weights]
    },
    scoring='f1',
    cv=3
)
grid_result = gsc.fit(X, y)

# Print the best parameters found by the grid search
print("Best parameters : %s" % grid_result.best_params_)

# Plot the weights vs f1 score for the grid search results
dataz = pd.DataFrame({ 'score': grid_result.cv_results_['mean_test_score'],
                       'weight': weights })
dataz.plot(x='weight')

# Create a RandomForestClassifier with balanced class weights
clf = RandomForestClassifier(criterion='entropy', n_jobs=-1, class_weight="balanced")
# Fit the RandomForestClassifier to the training data
clf.fit(X_train, y_train)

# Predict labels for the test set using the trained RandomForestClassifier
y_pred = clf.predict(X_test)

# Calculate and print the accuracy of the RandomForestClassifier
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
