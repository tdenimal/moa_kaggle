from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import random
import sys
from sklearn.metrics import log_loss

from sklearn.decomposition import PCA

import numpy as np # linear algebra
import pandas as pd 

import sys
sys.path.append('../input/iterativestratification')
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

data_dir = '../input/lish-moa'
#data_dir = '../data/01_raw'
os.listdir(data_dir)


# Parameters
no_ctl = True
scale = "rankgauss"
decompo = "PCA"
#ncompo_genes = 80
#ncompo_cells = 10
ncompo_genes = 600
ncompo_cells = 50
encoding = "dummy"
variance_threshold = .8


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


#RankGauss
for col in (GENES + CELLS):
    transformer = QuantileTransformer(n_quantiles=100,random_state=seed, output_distribution="normal")
    vec_len = len(data_all.loc[:,col].values)
    raw_vec = data_all.loc[:,col].values.reshape(vec_len, 1)
    transformer.fit(raw_vec)
    data_all.loc[:,col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]


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
    data_all = pd.get_dummies(data_all, columns = ["cp_time", "cp_dose"])


#for stats in ["sum", "mean", "std", "kurt", "skew"]:
for stats in ["sum",  "std"]:
    data_all["g_" + stats] = getattr(data_all[GENES], stats)(axis = 1)
    data_all["c_" + stats] = getattr(data_all[CELLS], stats)(axis = 1)

#for stats in ["sum", "mean", "std", "kurt", "skew"]:
for stats in ["sum"]:
    data_all["gc_" + stats] = getattr(data_all[GENES + CELLS], stats)(axis = 1)



from sklearn.feature_selection import VarianceThreshold

feat_cols = [c for c in data_all.columns if c not in ["sig_id", "cp_type"]]
var_thresh = VarianceThreshold(variance_threshold)  #<-- Update
data_feats = pd.DataFrame(var_thresh.fit_transform(data_all[feat_cols]))


data_all = pd.concat([data_all["sig_id"], data_feats],axis=1)


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


train_df.to_csv("train_v3.csv.gz",index=False,compression="gzip")
test_df.to_csv("test_v3.csv.gz",index=False,compression="gzip")


