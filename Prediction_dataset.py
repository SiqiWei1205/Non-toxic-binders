#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import pearsonr, mannwhitneyu
import statistics
from scipy import stats

# Read data from TSV files and rename columns for easier interpretation
data=pd.read_csv('CAIX_test.tsv', sep='\t')
data.rename(columns={'Ligand SMILES':'smiles'}, inplace=True)
data.to_csv('CAIX_test.csv')

data=pd.read_csv('seh_test_1.tsv', sep='\t')
data.rename(columns={'Ligand SMILES':'smiles'}, inplace=True)
data.to_csv('seh_test_1.csv')

seh=pd.read_csv('seh_preds_1.csv')
caix=pd.read_csv('caix_preds.csv')

# Plot histograms of the enrichment scores for SEH and CAIX
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

df1 = pd.DataFrame(dict(a=seh['enrichment_score']))
df2 = pd.DataFrame(dict(b=caix['enrichment_score']))

fig, axes = plt.subplots(1, 2)
df1.hist('a', ax=axes[0])
df2.hist('b', ax=axes[1])

plt.show()

# Plot overlapping histograms of the enrichment scores for SEH and CAIX
x = seh['enrichment_score']
y = caix['enrichment_score']
bins = np.linspace(0, 8, 100)

plt.hist(x, bins, alpha=0.5, label='Soluble Epoxide Hydrolase (SEH)')
plt.hist(y, bins, alpha=0.5, label='Carbonic Anhydrase IX (CAIX)')
plt.legend(loc='upper right')
plt.xlabel('Enrichment Score')
plt.ylabel('Count')
plt.show()

# Compute the Pearson correlation coefficient between CAIX Ki and enrichment score
corr, _ = pearsonr(caix['Ki (nM)'], caix['enrichment_score'])
print('Pearsons correlation: %.3f' % corr)

# Find common elements between two datasets based on SMILES column
def same_element(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return (set1 & set2), (set1 ^ set2), ((set1 | set2) - set2), ((set1 | set2) - set1)

same, dif, alone_forward, alone_backward = same_element(data1['smiles'], data2['smiles'])
print('Same elements:', same, 'number of same elements:', len(same))

# Compute descriptive statistics for CAIX and SEH enrichment scores
print('Median of CAIX:', np.median(caix['enrichment_score']))
print('Mean of CAIX:', np.mean(caix['enrichment_score']))
print('Mode of CAIX:', stats.mode(caix['enrichment_score']))
print('SD of CAIX:', statistics.stdev(caix['enrichment_score']))
print('Median of SEH:', np.median(seh['enrichment_score']))
print('Mean of SEH:', np.mean(seh['enrichment_score']))
print('Mode of SEH:', stats.mode(seh['enrichment_score']))
print('SD of SEH:', statistics.stdev(seh['enrichment_score']))

# Compute Mann-Whitney U test between CAIX and SEH enrichment scores
mannwhitneyu_result = mannwhitneyu(caix['enrichment_score'], seh['enrichment_score'])
print(mannwhitneyu_result)

# Plot histograms of the enrichment scores for CAIX, SEH, and multi-task dataset
df_1 = pd.read_csv('CAIX_preds_multi.csv')

x = seh['enrichment_score']
y = caix['enrichment_score']
z = df_1['enrichment_score']
bins = np.linspace(0, 8, 100)

plt.hist(x, bins, alpha=0.5, label='Ligands of Soluble Epoxide Hydrolase')
plt.hist(y, bins, alpha=0.5, label='Ligands of Carbonic Anhydrase IX (single-task)')
plt.hist(z, bins, alpha=0.5, label='Ligands of Carbonic Anhydrase IX (multi-task)')
plt.legend(loc='upper right')
plt.xlabel('Enrichment Score for CAIX')
plt.ylabel('Count')
plt.show()

# Compute Mann-Whitney U test between CAIX single-task and multi-task enrichment scores
mannwhitneyu_result_2 = mannwhitneyu(caix['enrichment_score'], df_1['enrichment_score'])
print(mannwhitneyu_result_2)
