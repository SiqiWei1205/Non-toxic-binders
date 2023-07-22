#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import sys
import os
import time
from typing import List
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.MolStandardize.rdMolStandardize import LargestFragmentChooser
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import hdbscan

# Function to get the largest fragment from SMILES string
def get_largest_fragment_from_smiles(s: str):
    mol = Chem.MolFromSmiles(s)
    if mol:
        clean_mol = LargestFragmentChooser().choose(mol)
        return Chem.MolToSmiles(clean_mol)
    return None

# Function to compute ECFP descriptors for a list of SMILES strings
def compute_ecfp_descriptors(smiles_list: List[str]):
    keep_idx = []
    descriptors = []
    for i, smiles in enumerate(smiles_list):
        ecfp = _compute_single_ecfp_descriptor(smiles)
        if ecfp is not None:
            keep_idx.append(i)
            descriptors.append(ecfp)

    return np.vstack(descriptors), keep_idx

# Function to compute a single ECFP descriptor for a given SMILES string
def _compute_single_ecfp_descriptor(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as E:
        return None

    if mol:
        fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        return np.array(fp)
    
    return None
 
# Function to plot global embeddings with clusters
def plot_global_embeddings_with_clusters(df: pd.DataFrame,
                          x_col: str,
                          y_col: str,
                          cluster_col: str,
                          title: str = "",
                          x_lim = None,
                          y_lim = None):
    clustered = df[cluster_col].values >= 0
    
    plt.figure(figsize=(10,8))
    ax = sns.scatterplot(data=df.iloc[~clustered],
                    x=x_col,
                    y=y_col,
                    color=(0.5, 0.5, 0.5),
                    s=10,
                    alpha=0.1)
    sns.scatterplot(data=df.iloc[clustered],
                    x=x_col,
                    y=y_col,
                    hue=cluster_col,
                    alpha=0.5,
                    palette="nipy_spectral",
                    ax=ax)
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)

    sm = plt.cm.ScalarMappable(cmap="nipy_spectral")
    sm.set_array([])
    ax.get_legend().remove()
    ax.figure.colorbar(sm, label="Global Cluster")
  
    plt.title(title)
    plt.show()

# Read in data from MoleculeNet
df = pd.read_csv("data_smile.csv")

# Clean up column names for easier interpretation
df = df[["smiles", "label"]].reset_index(drop=True)

# Compute descriptors and keep track of molecules that failed to featurize
ecfp_descriptors, keep_idx = compute_ecfp_descriptors(df["smiles"])

# Only keep molecules that successfully featurized
df = df.iloc[keep_idx]

# Time the UMAP embedding
umap_model = umap.UMAP(metric="jaccard",
                      n_neighbors=25,
                      n_components=2,
                      low_memory=False,
                      min_dist=0.001)
X_umap = umap_model.fit_transform(ecfp_descriptors)
df["UMAP_0"], df["UMAP_1"] = X_umap[:,0], X_umap[:,1]
     
# Time the PCA embedding
pca = PCA(n_components=2)
X_pca = pca.fit_transform(ecfp_descriptors)
df["PCA_0"], df["PCA_1"] = X_pca[:,0], X_pca[:,1]

# Time the t-SNE embedding
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(ecfp_descriptors)
df["TNSE_0"], df["TNSE_1"] = X_tsne[:,0], X_tsne[:,1]

# Define a palette for the plot
palette = sns.color_palette(["hotpink", "dodgerblue"])

# Plot each embedding with clusters colored by permeability
for method in ["UMAP", "PCA", "TNSE"]:
    plt.figure(figsize=(8,8))
    sns.scatterplot(data=df,
                    x=f"{method}_0",
                    y=f"{method}_1",
                    hue="permeable",
                    alpha=0.5,
                    palette=palette,
                    size="weight",
                    sizes=(40, 400),)
    plt.title(f"{method} Embedding of Dataset")
    plt.show()
