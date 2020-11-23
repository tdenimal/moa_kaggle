from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import random
import sys
from sklearn.metrics import log_loss

from sklearn.decomposition import PCA

import numpy as np # linear algebra
import pandas as pd 

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import QuantileTransformer

import numpy as np
from scipy.sparse import csc_matrix
import time
from abc import abstractmethod

from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import zipfile

data_dir = '../data/01_raw'
os.listdir(data_dir)


# Parameters
no_ctl = True
scale = "rankgauss"
decompo = "PCA"
#ncompo_genes = 80
#ncompo_cells = 10
ncompo_genes = 300
ncompo_cells = 25
encoding = "dummy"
variance_threshold = .7


#load data
train_features = pd.read_csv(data_dir+'/train_features.csv')
train_targets_scored = pd.read_csv(data_dir+'/train_targets_scored.csv')
train_targets_nonscored = pd.read_csv(data_dir+'/train_targets_nonscored.csv')

test_features = pd.read_csv(data_dir+'/test_features.csv')
drug = pd.read_csv(data_dir+'/train_drug.csv')

targets_scored = train_targets_scored.columns[1:]
targets_nscored = train_targets_nonscored.columns[1:]
scored = train_targets_scored.merge(drug, on='sig_id', how='left') 


seed = 42

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
seed_everything(seed)

data_all = pd.concat([train_features, test_features], ignore_index = True)

cols_numeric = [feat for feat in list(data_all.columns) if feat not in ["sig_id", "cp_type", "cp_time", "cp_dose"] +\
                                                                       list(train_targets_nonscored.columns)]

GENES = [col for col in data_all.columns if col.startswith("g-")]
CELLS = [col for col in data_all.columns if col.startswith("c-")]


#True gauss rank
import cupy as cp
from cupyx.scipy.special import erfinv
epsilon = 1e-6

for k in (cols_numeric):
    r_gpu = cp.array(data_all.loc[:,k])
    r_gpu = r_gpu.argsort().argsort()
    r_gpu = (r_gpu/r_gpu.max()-0.5)*2 
    r_gpu = cp.clip(r_gpu,-1+epsilon,1-epsilon)
    r_gpu = erfinv(r_gpu) 
    data_all.loc[:,k] = cp.asnumpy( r_gpu * np.sqrt(2) )




#PCA
if decompo == "PCA":
    print("PCA")
    
    pca_genes = PCA(n_components = ncompo_genes,
                    random_state = seed).fit_transform(data_all[GENES])
    pca_cells = PCA(n_components = ncompo_cells,
                    random_state = seed).fit_transform(data_all[CELLS])
    
    pca_genes = pd.DataFrame(pca_genes, columns = [f"pca_g-{i}" for i in range(ncompo_genes)])
    pca_cells = pd.DataFrame(pca_cells, columns = [f"pca_c-{i}" for i in range(ncompo_cells)])
    data_all = pd.concat([data_all, pca_genes, pca_cells], axis = 1)
else:
    pass


# Encoding
if encoding == "lb":
    print("Label Encoding")
    for feat in ["cp_time", "cp_dose"]:
        data_all[feat] = LabelEncoder().fit_transform(data_all[feat])
elif encoding == "dummy":
    print("One-Hot")
    dummies = ['cp_time_24', 'cp_time_48', 'cp_time_72', 'cp_dose_D1', 'cp_dose_D2']
    df_dummies = pd.get_dummies(data_all, columns = ["cp_time", "cp_dose"])[dummies]


for stats in ["sum", "mean", "std", "kurt", "skew"]:
#for stats in ["sum",  "std"]:
    data_all["g_" + stats] = getattr(data_all[GENES], stats)(axis = 1)
    data_all["c_" + stats] = getattr(data_all[CELLS], stats)(axis = 1)

for stats in ["sum", "mean", "std", "kurt", "skew"]:
#for stats in ["sum"]:
    data_all["gc_" + stats] = getattr(data_all[GENES + CELLS], stats)(axis = 1)

def fe_stats(df):
    
    features_g = GENES
    features_c = CELLS
    
    gsquarecols=['g-574','g-211','g-216','g-0','g-255','g-577','g-153','g-389','g-60','g-370','g-248','g-167','g-203','g-177','g-301','g-332','g-517','g-6','g-744','g-224','g-162','g-3','g-736','g-486','g-283','g-22','g-359','g-361','g-440','g-335','g-106','g-307','g-745','g-146','g-416','g-298','g-666','g-91','g-17','g-549','g-145','g-157','g-768','g-568','g-396']
        
    df['c52_c42'] = df['c-52'] * df['c-42']
    df['c13_c73'] = df['c-13'] * df['c-73']
    df['c26_c13'] = df['c-23'] * df['c-13']
    df['c33_c6'] = df['c-33'] * df['c-6']
    df['c11_c55'] = df['c-11'] * df['c-55']
    df['c38_c63'] = df['c-38'] * df['c-63']
    df['c38_c94'] = df['c-38'] * df['c-94']
    df['c13_c94'] = df['c-13'] * df['c-94']
    df['c4_c52'] = df['c-4'] * df['c-52']
    df['c4_c42'] = df['c-4'] * df['c-42']
    df['c13_c38'] = df['c-13'] * df['c-38']
    df['c55_c2'] = df['c-55'] * df['c-2']
    df['c55_c4'] = df['c-55'] * df['c-4']
    df['c4_c13'] = df['c-4'] * df['c-13']
    df['c82_c42'] = df['c-82'] * df['c-42']
    df['c66_c42'] = df['c-66'] * df['c-42']
    df['c6_c38'] = df['c-6'] * df['c-38']
    df['c2_c13'] = df['c-2'] * df['c-13']
    df['c62_c42'] = df['c-62'] * df['c-42']
    df['c90_c55'] = df['c-90'] * df['c-55']
        
        
    for feature in features_c:
        df[f'{feature}_squared'] = df[feature] ** 2     
                
    for feature in gsquarecols:
        df[f'{feature}_squared'] = df[feature] ** 2        
        
    return df

data_all=fe_stats(data_all)


from sklearn.feature_selection import VarianceThreshold

feat_cols = [c for c in data_all.columns if c not in ["sig_id", "cp_type","cp_time","cp_dose"]]
var_thresh = VarianceThreshold(variance_threshold)  #<-- Update
data_feats = pd.DataFrame(var_thresh.fit_transform(data_all[feat_cols]))



data_all = pd.concat([data_all["sig_id"],df_dummies, data_feats],axis=1)


print(data_all.shape)


train_df = data_all[: train_features.shape[0]]
train_df.reset_index(drop = True, inplace = True)
# The following line it's a bad practice in my opinion, targets on train set
#train_df = pd.concat([train_df, targets], axis = 1)
test_df = data_all[train_df.shape[0]: ]
test_df.reset_index(drop = True, inplace = True)

del data_all

print(f"train_df.shape: {train_df.shape}")
print(f"test_df.shape: {test_df.shape}")


NFOLDS = 5


# LOCATE DRUGS
vc = scored.drug_id.value_counts()
vc1 = vc.loc[vc<=18].index.sort_values()
vc2 = vc.loc[vc>18].index.sort_values()

# STRATIFY DRUGS 18X OR LESS
dct1 = {}; dct2 = {}
skf = MultilabelStratifiedKFold(n_splits=NFOLDS, shuffle=True, 
          random_state=seed)
tmp = scored.groupby('drug_id')[targets_scored].mean().loc[vc1]
for fold,(idxT,idxV) in enumerate( skf.split(tmp,tmp[targets_scored])):
    dd = {k:fold for k in tmp.index[idxV].values}
    dct1.update(dd)

# STRATIFY DRUGS MORE THAN 18X
skf = MultilabelStratifiedKFold(n_splits=NFOLDS, shuffle=True, 
          random_state=seed)
tmp = scored.loc[scored.drug_id.isin(vc2)].reset_index(drop=True)
for fold,(idxT,idxV) in enumerate( skf.split(tmp,tmp[targets_scored])):
    dd = {k:fold for k in tmp.sig_id[idxV].values}
    dct2.update(dd)

# ASSIGN FOLDS
train_df = train_df.merge(drug,on="sig_id")
train_df['fold'] = train_df.drug_id.map(dct1)
train_df.loc[train_df.fold.isna(),'fold'] =\
    train_df.loc[train_df.fold.isna(),'sig_id'].map(dct2)
train_df.fold = train_df.fold.astype('int8')


train_df.to_csv("train_v1c.csv.gz",index=False,compression="gzip")
test_df.to_csv("test_v1c.csv.gz",index=False,compression="gzip")


