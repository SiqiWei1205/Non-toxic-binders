#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import chemprop

# Read data from CSV files for toxicity and enrichment data
df_t = pd.read_csv('toxicology.csv')
df_t.rename({'SMILES': 'smiles'}, axis=1, inplace=True)
df_e = pd.read_csv('data_score.csv',index_col=0)
# Remove duplicate rows based on the 'smiles' column to keep unique molecules
df_t = df_t.drop_duplicates(subset=['smiles'])
df_e = df_e.drop_duplicates(subset=['smiles'])

# Merge dataframes based on the 'smiles' column and perform an inner join to get common molecules
output1 = pd.merge(df_e, df_t, how='inner',on='smiles',validate='1:1')
output1.to_csv("multi.csv")

# using chemprop to predict enrichment and toxicity scores
arguments = [
    '--data_path', 'multi.csv',
    '--dataset_type', 'regression',
    '--save_dir', 'test_checkpoints_multi',
    '--epochs', '5',
    '--save_smiles_splits',
    '--number_of_molecules', '1',
    '--smiles_column', 'smiles',
    '--target_columns','enrichment_score', 'Acetylcholinesterase_Human_', 'CB1_', 'COX1_Human_Cyclooxygenase', 'COX2_Human_Cyclooxygenase', 'DAT_Human_Dopamine_Transporter', 'ETA_Human_Endothelin_GPCR', 'Glutamate_(NMDA,Non-Selective)_Rat_Ion_Channel', 'Human_Acetylcholine_(Muscarinic)_M1_GPCR', 'Human_Acetylcholine_(Muscarinic)_M2_GPCR', 'Human_Acetylcholine_(Muscarinic)_M3_GPCR', 'Human_Adenosine_receptor_A2A', 'Human_Adrenoceptor_alpha1A', 'Human_Adrenoceptor_alpha2A', 'Human_Adrenoceptor_beta1', 'Human_Adrenoceptor_beta2', 'Human_Androgen_NHR', 'Human_CB2', 'Human_Cholecystokinin_CCK1_(CCKA)', 'Human_Dopamine_D1', 'Human_Dopamine_D2S_(D2-short)', 'Human_Glucocorticoid_NHR', 'Human_H1_Histamine_GPCR', 'Human_H2_Histamine_GPCR', 'Human_Serotonin_5-HT1A', 'Human_Serotonin_5-HT1B', 'Human_Serotonin_5-HT2B', 'Human_Serotonin_5-HT3a', 'Human_Serotonin_Transporter', 'Human_delta_opioid_GPCR', 'Human_kappa_opioid_GPCR', 'Human_mu_opioid_GPCR', 'Lck_Human_TK_Kinase', 'Monoamine_Oxidase_A_(MAO-A)_Rat_Binding', 'NET_Human_Norepinephrine_Transporter', 'Non-Selective_Rat_GABAA_Ion_Channel_CHEMBL1907607', 'Non-Selective_Rat_Sodium_Ion_Channel', 'Non-Selective_Rat_Sodium_Ion_Channel_CHEMBL2095171', 'PDE3A2', 'PDE4D2', 'Serotonin_5-HT2A', 'Vasopressin_Oxytocin_GPCR', 'herg_CHEMBL240'
]
# Parse the arguments for training from chemprop.args.TrainArgs
args = chemprop.args.TrainArgs().parse_args(arguments)
# Perform cross-validation using chemprop.train.cross_validate
mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)

# Read the predictions from the CAIX_preds_multi.csv file
df = pd.read_csv('CAIX_preds_multi.csv')
# Calculate the mean toxicity score from the selected toxicity target columns
df['Mean_toxicity']=df[['Acetylcholinesterase_Human_', 'CB1_', 'COX1_Human_Cyclooxygenase', 'COX2_Human_Cyclooxygenase', 'DAT_Human_Dopamine_Transporter', 'ETA_Human_Endothelin_GPCR', 'Glutamate_(NMDA,Non-Selective)_Rat_Ion_Channel', 'Human_Acetylcholine_(Muscarinic)_M1_GPCR', 'Human_Acetylcholine_(Muscarinic)_M2_GPCR', 'Human_Acetylcholine_(Muscarinic)_M3_GPCR', 'Human_Adenosine_receptor_A2A', 'Human_Adrenoceptor_alpha1A', 'Human_Adrenoceptor_alpha2A', 'Human_Adrenoceptor_beta1', 'Human_Adrenoceptor_beta2', 'Human_Androgen_NHR', 'Human_CB2', 'Human_Cholecystokinin_CCK1_(CCKA)', 'Human_Dopamine_D1', 'Human_Dopamine_D2S_(D2-short)', 'Human_Glucocorticoid_NHR', 'Human_H1_Histamine_GPCR', 'Human_H2_Histamine_GPCR', 'Human_Serotonin_5-HT1A', 'Human_Serotonin_5-HT1B', 'Human_Serotonin_5-HT2B', 'Human_Serotonin_5-HT3a', 'Human_Serotonin_Transporter', 'Human_delta_opioid_GPCR', 'Human_kappa_opioid_GPCR', 'Human_mu_opioid_GPCR', 'Lck_Human_TK_Kinase', 'Monoamine_Oxidase_A_(MAO-A)_Rat_Binding', 'NET_Human_Norepinephrine_Transporter', 'Non-Selective_Rat_GABAA_Ion_Channel_CHEMBL1907607', 'Non-Selective_Rat_Sodium_Ion_Channel', 'Non-Selective_Rat_Sodium_Ion_Channel_CHEMBL2095171', 'PDE3A2', 'PDE4D2', 'Serotonin_5-HT2A', 'Vasopressin_Oxytocin_GPCR', 'herg_CHEMBL240']].mean(axis=1)
# Sort the dataframes based on mean toxicity and enrichment scores
df_toxic=df.sort_values(by='Mean_toxicity',inplace=False)
df_enrich = df.sort_values(by='enrichment_score',inplace=False,ascending=True)

import numpy as np
data1 = df_toxic
data2 = df_enrich
# Function to find the common, different, and unique elements between two lists
def same_element(list1,list2):
    set1 = set(list1)
    set2 = set(list2)
    return (set1 & set2),(set1 ^ set2),((set1|set2)-set2),((set1|set2)-set1)
# Find common, different, and unique smiles between toxicity and enrichment data
same,dif,alone_forward,alone_backward = same_element(data1['smiles'],data2['smiles'])
print('Same elements：',same,'number of same elements：',len(same))
# Extract mean toxicity and enrichment scores and save them to CSV file
Mean_t=df_toxic['Mean_toxicity']

Mean_t.to_csv("Mean_toxicity.csv")

enrich=df_enrich['enrichment_score']
enrich.to_csv("enrichment.csv")
