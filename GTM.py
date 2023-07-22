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
