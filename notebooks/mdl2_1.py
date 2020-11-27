#import numpy as np
import cupy as cp
#import torch.cuda.nvtx as nvtx

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import random
import sys
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from sklearn.metrics import log_loss


import pandas as pd 

import torch
from scipy.sparse import csc_matrix
import time
from abc import abstractmethod

from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score
from torch.nn.utils import clip_grad_norm_

from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader
from copy import deepcopy
#import io
#import json
#from pathlib import Path
#import shutil
#import zipfile

import torch
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F


#data_dir = '../input/lish-moa'
data_dir = '../data/01_raw'
os.listdir(data_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"

#load data
train_targets_scored = pd.read_csv(data_dir+'/train_targets_scored.csv')
train_targets_nonscored = pd.read_csv(data_dir+'/train_targets_nonscored.csv')
targets_scored = train_targets_scored.columns[1:]
targets_nscored = train_targets_nonscored.columns[1:]

seed = 42
NFOLDS = 5

# HyperParameters
EPOCHS = 30 #30
PATIENCE=40

BATCH_SIZE = 128
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 1e-5        
EARLY_STOPPING_STEPS = PATIENCE+5
EARLY_STOP = False

hidden_sizes = [1200,1000,1000] #[1200,1000,1000]
dropout_rates = [0.2619422201258426,0.2,0.2]  #[0.2619422201258426,0.2619422201258426,0.27]
#SEED = [0,1,2,3,4,5,6] #<-- Update
#SEED = [0,3,6]
SEED = [0]
pct_start=0.1
div_factor=1e4 
final_div_factor=1e3
max_lr=1e-2



def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
seed_everything(seed)

train_df = pd.read_csv("train_v3.csv.gz",compression="gzip")
test_df = pd.read_csv("test_v3.csv.gz",compression="gzip")


feat_cols = [c for c in train_df.columns if c not in ["sig_id","cp_type","drug_id","fold"]]
num_features=len(feat_cols)
num_targets_scored=len(targets_scored)
num_targets_nscored=len(targets_nscored)



class TrainDataset:
    def __init__(self, features, targets_scored,targets_nscored):
        self.features = features
        self.targets_scored = targets_scored
        self.targets_nscored = targets_nscored
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float),
            'y_scored' : torch.tensor(self.targets_scored[idx, :], dtype=torch.float),
            'y_nscored' : torch.tensor(self.targets_nscored[idx, :], dtype=torch.float)           
        }
        return dct

class ValidDataset:
    def __init__(self, features, targets_scored):
        self.features = features
        self.targets_scored = targets_scored
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float),
            'y_scored' : torch.tensor(self.targets_scored[idx, :], dtype=torch.float),          
        }
        return dct
    
class TestDataset:
    def __init__(self, features):
        self.features = features
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float)
        }
        return dct

def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0
    

    for batch_idx, data in enumerate(dataloader):
        #nvtx.range_push("Batch " + str(batch_idx))

        #nvtx.range_push("Copy to Device")
        inputs, targets1, targets2 = data['x'].to(device), data['y_scored'].to(device), data['y_nscored'].to(device)
#         print(inputs.shape)
        #nvtx.range.pop()

        #nvtx.range_push("Forward pass")
        optimizer.zero_grad()
        outputs1,outputs2 = model(inputs)
        loss1 = loss_fn(outputs1, targets1)
        loss2 = loss_fn(outputs2, targets2)
        loss = loss1 + loss2
        #nvtx.range.pop()
        #nvtx.range_push("Backward pass")
        loss.backward()
        optimizer.step()
        scheduler.step()
        #nvtx.range.pop()
        #nvtx.range.pop()
        final_loss += loss.item()
        
    final_loss /= len(dataloader)
    
    return final_loss


def valid_fn(model, loss_fn, dataloader, device):
    model.eval()
    final_loss = 0
    valid_preds = []
    
    for data in dataloader:
        inputs, targets = data['x'].to(device), data['y_scored'].to(device)
        outputs,_ = model(inputs)
        loss = loss_fn(outputs, targets)
        
        final_loss += loss.item()
        valid_preds.append(outputs.sigmoid().detach().cpu().numpy())
        

    final_loss /= len(dataloader)


    valid_preds = np.concatenate(valid_preds)
    
    return final_loss, valid_preds

def inference_fn(model, dataloader, device):
    model.eval()
    preds_scored = []
    
    for data in dataloader:
        inputs = data['x'].to(device)

        with torch.no_grad():
            outputs_scored,_ = model(inputs)
        
        preds_scored.append(outputs_scored.sigmoid().detach().cpu().numpy())
        
    preds_scored = np.concatenate(preds_scored)
    
    return preds_scored



import torch
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
            self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss


class Model(nn.Module):
    def __init__(self, num_features, 
                 num_targets_scored,
                 num_targets_nscored, 
                 hidden_sizes,
                 dropout_rates):

        super(Model, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_sizes[0]))
        self.activation1 = torch.nn.PReLU(num_parameters = hidden_sizes[0], init = 1.0)
        
        self.batch_norm2 = nn.BatchNorm1d(hidden_sizes[0])
        self.dropout2 = nn.Dropout(dropout_rates[0])
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_sizes[0], hidden_sizes[1]))
        self.activation2 = torch.nn.PReLU(num_parameters = hidden_sizes[1], init = 1.0)


        self.batch_norm2b = nn.BatchNorm1d(hidden_sizes[1])
        self.dropout2b = nn.Dropout(dropout_rates[1])
        self.dense2b = nn.utils.weight_norm(nn.Linear(hidden_sizes[1], hidden_sizes[2]))
        self.activation2b = torch.nn.PReLU(num_parameters = hidden_sizes[2], init = 1.0)
        
        self.batch_norm3 = nn.BatchNorm1d(hidden_sizes[2])
        self.dropout3 = nn.Dropout(dropout_rates[2])
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_sizes[2], num_targets_scored))
        self.dense4 = nn.utils.weight_norm(nn.Linear(hidden_sizes[2], num_targets_nscored))

    def init_bias(self,pos_scored_rate,pos_nscored_rate):
        self.dense3.bias.data = nn.Parameter(torch.tensor(pos_scored_rate, dtype=torch.float))
        self.dense4.bias.data = nn.Parameter(torch.tensor(pos_nscored_rate, dtype=torch.float))
    
    def recalibrate_layer(self, layer):
        if(torch.isnan(layer.weight_v).sum() > 0):
            print ('recalibrate layer.weight_v')
            layer.weight_v = torch.nn.Parameter(torch.where(torch.isnan(layer.weight_v), torch.zeros_like(layer.weight_v), layer.weight_v))
            layer.weight_v = torch.nn.Parameter(layer.weight_v + 1e-7)

        if(torch.isnan(layer.weight).sum() > 0):
            print ('recalibrate layer.weight')
            layer.weight = torch.where(torch.isnan(layer.weight), torch.zeros_like(layer.weight), layer.weight)
            layer.weight += 1e-7

    def forward(self, x):
        x = self.batch_norm1(x)
        self.recalibrate_layer(self.dense1)
        x = self.activation1(self.dense1(x))
        
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        self.recalibrate_layer(self.dense2)
        x = self.activation2(self.dense2(x))

        x = self.batch_norm2b(x)
        x = self.dropout2b(x)
        self.recalibrate_layer(self.dense2b)
        x = self.activation2b(self.dense2b(x))
        
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        self.recalibrate_layer(self.dense3)
        x1 = self.dense3(x)

        self.recalibrate_layer(self.dense4)
        x2 = self.dense4(x)
        return x1,x2
    
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))





num_features=len(feat_cols)
num_targets_scored=len(targets_scored)
num_targets_nscored=len(targets_nscored)




def run_training(fold, seed):
    
    seed_everything(seed)
    
    train = train_df[train_df['fold'] != fold][feat_cols]
    valid = train_df[train_df['fold'] == fold][feat_cols]


    X_train, y_scored_train,y_nscored_train   = train.values, train_targets_scored.iloc[:,1:].values[train.index, :], train_targets_nonscored.iloc[:,1:].values[train.index, :]
    X_val, y_val = valid.values, train_targets_scored.iloc[:,1:].values[valid.index, :]
    
    train_dataset = TrainDataset(X_train, y_scored_train, y_nscored_train)
    valid_dataset = ValidDataset(X_val, y_val)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Model(
        num_features=num_features,
        num_targets_scored=num_targets_scored,
        num_targets_nscored=num_targets_nscored,
        hidden_sizes=hidden_sizes,
        dropout_rates=dropout_rates,
    )

    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                              pct_start=pct_start,
                                              div_factor=div_factor,
                                              final_div_factor=final_div_factor,
                                              max_lr=max_lr,
                                              epochs=EPOCHS, steps_per_epoch=len(trainloader))


    
    loss_fn = nn.BCEWithLogitsLoss()
    loss_tr = SmoothBCEwLogits(smoothing =0.001)
    
    early_stopping_steps = EARLY_STOPPING_STEPS
    early_step = 0
   
    oof = np.zeros((train_df.shape[0], num_targets_scored))
    best_loss = np.inf
    
    for epoch in range(EPOCHS):
        
        train_loss = train_fn(model, optimizer,scheduler, loss_tr, trainloader, device)
        valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, device)
       # print(f"FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss}, valid_loss: {valid_loss}")
        print(f"SEED: {seed}, FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss:.6f}, valid_loss: {valid_loss:.6f}")
        #scheduler.step(valid_loss)
        
        if valid_loss < best_loss:
            
            best_loss = valid_loss
            oof[valid.index] = valid_preds
            torch.save(model.state_dict(), f"FOLD{fold}_m1_1.pth")
        
        elif(EARLY_STOP == True):
            
            early_step += 1
            if (early_step >= early_stopping_steps):
                break
            
    
    #--------------------- PREDICTION---------------------
    x_test = test_df[feat_cols].values
    testdataset = TestDataset(x_test)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Model(
        num_features=num_features,
        num_targets_scored=num_targets_scored,
        num_targets_nscored=num_targets_nscored,
        hidden_sizes=hidden_sizes,
        dropout_rates=dropout_rates
    )
    
    model.load_state_dict(torch.load(f"FOLD{fold}_m1_1.pth"))
    model.to(device)
    
    predictions_scored = np.zeros((test_df.shape[0], num_targets_scored))
    predictions_scored = inference_fn(model, testloader, device)
    
    return oof, predictions_scored


def run_k_fold(NFOLDS, seed):
    oof = np.zeros((train_df.shape[0], num_targets_scored))
    predictions_scored = np.zeros((test_df.shape[0], num_targets_scored))
    
    for fold in range(NFOLDS):
        oof_, pred_scored_ = run_training(fold, seed)
        
        predictions_scored += pred_scored_ / NFOLDS
        oof += oof_
        
    return oof, predictions_scored


# Averaging on multiple SEEDS


oof = np.zeros((train_df.shape[0], num_targets_scored))
predictions_scored = np.zeros((test_df.shape[0], num_targets_scored))

for seed in SEED:
    
    oof_, predictions_scored_ = run_k_fold(NFOLDS, seed)
    oof += oof_ / len(SEED)
    predictions_scored += predictions_scored_ / len(SEED)

train_df[targets_scored] = oof
test_df[targets_scored] = predictions_scored



y_true = train_targets_scored.iloc[:,1:].values
y_pred = train_df[targets_scored].values

score = 0
for i in range(len(targets_scored)):
    score_ = log_loss(y_true[:, i], y_pred[:, i])
    score += score_ / len(targets_scored)
    
print("CV log_loss: ", score)


from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_true,y_pred))



train_features = pd.read_csv(data_dir+'/train_features.csv')
sample_submission = pd.read_csv(data_dir+'/sample_submission.csv')

oof = train_features[["sig_id"]].merge(train_df[train_targets_scored.columns], on='sig_id', how='inner')
sub = sample_submission.drop(columns=targets_scored).merge(test_df[train_targets_scored.columns], on='sig_id', how='left').fillna(0)

oof.to_csv("oof_v2_1.csv.gz",index=False,compression="gzip")
sub.to_csv("sub_v2_1.csv.gz",index=False,compression="gzip")
