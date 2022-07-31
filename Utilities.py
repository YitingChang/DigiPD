#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import os
import numpy as np
import scipy.stats as stats
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.utils import resample

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


# In[9]:


# Upsampling
def upsampling(df_score, df_feature, df_meta, upsample_size, counts, score):
    upsampled_df_score = pd.DataFrame()
    upsampled_df_feature = pd.DataFrame()
    upsampled_df_meta = pd.DataFrame()

    for i,count in enumerate(counts):
        if count == max(counts):
            upsampled_score_group = df_score.loc[score==i]
            upsampled_feature_group = df_feature.loc[score==i]
            upsampled_meta_group = df_meta.loc[score==i]    
        else:
            df_score_group = df_score.loc[score==i]
            df_feature_group = df_feature.loc[score==i]
            df_meta_group = df_meta.loc[score==i]
            idx_list = df_feature_group.index.tolist()
            resample_idx = resample(idx_list, replace=True, n_samples=upsample_size, random_state=27)
            upsampled_score_group = df_score_group.loc[resample_idx]
            upsampled_feature_group = df_feature_group.loc[resample_idx]
            upsampled_meta_group = df_meta_group.loc[resample_idx]
            
        upsampled_df_score = pd.concat([upsampled_df_score, upsampled_score_group])
        upsampled_df_feature = pd.concat([upsampled_df_feature, upsampled_feature_group])
        upsampled_df_meta = pd.concat([upsampled_df_meta, upsampled_meta_group])

    upsampled_df_score = upsampled_df_score.reset_index(drop=True)
    upsampled_df_feature = upsampled_df_feature.reset_index(drop=True)
    upsampled_df_meta = upsampled_df_meta.reset_index(drop=True)

    return upsampled_df_score, upsampled_df_feature, upsampled_df_meta

# Try other upsmapling methods:
# from imblearn.over_sampling import SMOTE


# In[7]:


# Get train/validation/test datasets
# df: data frame containing selected features
# score: classification label
# subject_id: the subject id column of data frame containing classification label (score)
def train_val_test_split(df, score, subject_id):
    # train:validation:test = 0.5:0.25:0.25
    sb_train = ['6_BOS', '16_BOS', '7_NYC', '14_BOS', '8_NYC', '5_BOS', '12_NYC', '6_NYC', '17_BOS',
               '4_BOS', '11_BOS', '10_BOS', '15_BOS', '4_NYC', '11_NYC']
    sb_val = ['8_BOS', '18_BOS', '2_NYC', '9_NYC', '3_BOS', '9_BOS']
    sb_test = ['19_BOS', '3_NYC', '7_BOS', '5_NYC', '13_BOS', '10_NYC', '12_BOS']

    # Train/Validation/Test Split
    is_train = subject_id.isin(sb_train).tolist()
    is_val = subject_id.isin(sb_val).tolist()
    is_test = subject_id.isin(sb_test).tolist()
    is_train_val = subject_id.isin(sb_train) | subject_id.isin(sb_val)
    is_train_val = is_train_val.tolist()

    X_train_valid = df.loc[is_train_val]
    y_train_valid = score[is_train_val]
    X_train = df.loc[is_train]
    y_train = score[is_train]
    X_valid = df.loc[is_val]
    y_valid = score[is_val]
    X_test = df.loc[is_test]
    y_test = score[is_test]
    
    return X_train_valid, y_train_valid, X_train, y_train,  X_valid, y_valid, X_test, y_test


# In[2]:


# Removing columns with zero variance in a panda dataframe using sklearn- VarianceThreshold
def pdVarianceThreshold(df, varThreshold):
    sel = VarianceThreshold(threshold=varThreshold)
    new_df = sel.fit_transform(df)
    new_filter = sel.get_support()
    feature_names = df.columns
    new_feature_names = feature_names[new_filter]
    new_df = pd.DataFrame(new_df, columns=new_feature_names)
    return new_df


# In[3]:


# Select columns with relevant features in a panda dataframe using sklearn- Univariate Selection
def pdSelectKBest(df, score, score_function, k_num):
    test = SelectKBest(score_func=score_function, k=k_num)
    new_df = test.fit_transform(df, score)
    new_filter = test.get_support()
    feature_names = df.columns
    new_feature_names = feature_names[new_filter]
    new_df = pd.DataFrame(new_df, columns=new_feature_names)
    return new_df


# In[10]:


# Select best classifier based on recall threshold for the positive class and F1 score
def SelectBestClf(valid_scores, recall_1_threshold, clf_best_params):
    is_good_recall_1 = valid_scores['Recall_1'] >= recall_1_threshold
    if sum(is_good_recall_1) == 0: # recall 1 is below the threshold 
        best_recall_1 = valid_scores['Recall_1'].max()
        is_best = valid_scores['Recall_1'] == best_recall_1
        if sum(is_best)>1: # duplicate max recall scores -> compare F1 score
            max_index = valid_scores['F1_micro'].loc[is_best].idxmax()
        else:
            max_index = valid_scores['Recall_1'].idxmax() 
        best_F1_micro = valid_scores['F1_micro'].iloc[max_index]

    else: # recall 1 is above the threshold -> select the best classifier based on F1 score
        best_F1_micro = valid_scores['F1_micro'].loc[is_good_recall_1].max()
        is_best = valid_scores['F1_micro'] == best_F1_micro
        if sum(is_best)>1: # duplicate max F1 scores -> compare recall score
            max_index = valid_scores['Recall_1'].loc[is_best].idxmax()
        else:
            max_index = [index for index, element in enumerate(is_best) if element] 
            max_index = max_index[0]
        best_recall_1 = valid_scores['Recall_1'].iloc[max_index]

    best_clf = valid_scores['Classifer'].iloc[max_index]
    training_time = valid_scores['Training time'].iloc[max_index]
    best_clf_params = clf_best_params[best_clf] 

    return best_F1_micro, best_recall_1, best_clf, best_clf_params 


# In[ ]:




