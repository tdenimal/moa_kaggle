import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

import os
import random
import sys
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from sklearn.metrics import log_loss

import pandas as pd 

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
ALL_TARGETS_EPOCHS = 30 #30
SCORED_ONLY_EPOCHS = 40 #30

BATCH_SIZE = 128
WEIGHT_DECAY = {'ALL_TARGETS': 1e-5, 'SCORED_ONLY': 3e-6}
MAX_LR = {'ALL_TARGETS': 1e-2, 'SCORED_ONLY': 3e-3}
DIV_FACTOR = {'ALL_TARGETS': 1e3, 'SCORED_ONLY': 1e2}
FINAL_DIV_FACTOR = {'ALL_TARGETS': 1e3, 'SCORED_ONLY': 1e3} #change
PCT_START = 0.1


#hidden_sizes = [1500,1250,1000,750] #[1500,1250,1000,750]
hidden_sizes = [1500,1200,1000,1000]
dropout_rates = [0.4, 0.25, 0.25, 0.25] #[0.5, 0.35, 0.3, 0.25]
#SEED = [0,1,2,3,4,5,6] #<-- Update
#SEED = [0,3,6]
SEED = [0]



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



class MoADataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
           'x' : torch.tensor(self.features[idx, :], dtype=torch.float),
           'y' : torch.tensor(self.targets[idx, :], dtype=torch.float),         
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
    
    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        final_loss += loss.item()
        
    final_loss /= len(dataloader)
    
    return final_loss


def valid_fn(model, loss_fn, dataloader, device):
    model.eval()
    final_loss = 0
    valid_preds = []
    
    for data in dataloader:
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
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
            outputs_scored = model(inputs)
        
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
                 num_targets,
                 ):

        super(Model, self).__init__()

        self.hidden_sizes = hidden_sizes
        self.dropout_rates = dropout_rates

        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.Linear(num_features, self.hidden_sizes[0])
        
        self.batch_norm2 = nn.BatchNorm1d(self.hidden_sizes[0])
        self.dropout2 = nn.Dropout(self.dropout_rates[0])
        self.dense2 = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])

        self.batch_norm3 = nn.BatchNorm1d(self.hidden_sizes[1])
        self.dropout3 = nn.Dropout(self.dropout_rates[1])
        self.dense3 = nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2])

        self.batch_norm4 = nn.BatchNorm1d(self.hidden_sizes[2])
        self.dropout4 = nn.Dropout(self.dropout_rates[2])
        self.dense4 = nn.Linear(self.hidden_sizes[2], self.hidden_sizes[3])

        self.batch_norm5 = nn.BatchNorm1d(self.hidden_sizes[3])
        self.dropout5 = nn.Dropout(self.dropout_rates[3])
        self.dense5 = nn.utils.weight_norm(nn.Linear(self.hidden_sizes[3], num_targets))


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
        x = F.leaky_relu(self.dense1(x))
        
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = F.leaky_relu(self.dense3(x))

        x = self.batch_norm4(x)
        x = self.dropout4(x)
        x = F.leaky_relu(self.dense4(x))

        x = self.batch_norm5(x)
        x = self.dropout5(x)
        x = self.dense5(x)
        return x
    
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


class FineTuneScheduler:
    def __init__(self, epochs):
        self.epochs = epochs
        self.epochs_per_step = 0
        self.frozen_layers = []

    def copy_without_top(self, model, num_features, num_targets, num_targets_new):
        self.frozen_layers = []

        model_new = Model(num_features, num_targets)
        model_new.load_state_dict(model.state_dict())

        # Freeze all weights
        for name, param in model_new.named_parameters():
            layer_index = name.split('.')[0][-1]

            if layer_index == 5:
                continue

            param.requires_grad = False

            # Save frozen layer names
            if layer_index not in self.frozen_layers:
                self.frozen_layers.append(layer_index)

        self.epochs_per_step = self.epochs // len(self.frozen_layers)

        # Replace the top layers with another ones
        model_new.batch_norm5 = nn.BatchNorm1d(model_new.hidden_sizes[3])
        model_new.dropout5 = nn.Dropout(model_new.dropout_rates[3])
        model_new.dense5 = nn.utils.weight_norm(nn.Linear(model_new.hidden_sizes[-1], num_targets_new))
        model_new.to(device)
        return model_new

    def step(self, epoch, model):
        if len(self.frozen_layers) == 0:
            return

        if epoch % self.epochs_per_step == 0:
            last_frozen_index = self.frozen_layers[-1]
            
            # Unfreeze parameters of the last frozen layer
            for name, param in model.named_parameters():
                layer_index = name.split('.')[0][-1]

                if layer_index == last_frozen_index:
                    param.requires_grad = True

            del self.frozen_layers[-1]  # Remove the last layer as unfrozen




def run_training(fold, seed):
    
    seed_everything(seed)
    
    train = train_df[train_df['fold'] != fold][feat_cols]
    valid = train_df[train_df['fold'] == fold][feat_cols]


    X_train, y_scored_train,y_nscored_train   = train.values, train_targets_scored.iloc[:,1:].values[train.index, :], train_targets_nonscored.iloc[:,1:].values[train.index, :]
    X_val, y_scored_val, y_nscored_val = valid.values, train_targets_scored.iloc[:,1:].values[valid.index, :],train_targets_nonscored.iloc[:,1:].values[valid.index, :]
    

    def train_model(model, tag_name, fine_tune_scheduler=None):


        if tag_name == 'ALL_TARGETS':
            y_train = np.append(y_scored_train,y_nscored_train,axis=1)
            y_val = np.append(y_scored_val,y_nscored_val,axis=1)
            oof = np.zeros((train_df.shape[0], num_targets_scored+num_targets_nscored))
            EPOCHS = ALL_TARGETS_EPOCHS

        else:
            y_train = y_scored_train
            y_val = y_scored_val
            oof = np.zeros((train_df.shape[0], num_targets_scored))
            EPOCHS = SCORED_ONLY_EPOCHS

        train_dataset = MoADataset(X_train, y_train)
        valid_dataset = MoADataset(X_val, y_val)


        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY[tag_name])
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                  steps_per_epoch=len(trainloader),
                                                  pct_start=PCT_START,
                                                  div_factor=DIV_FACTOR[tag_name],
                                                  final_div_factor=FINAL_DIV_FACTOR[tag_name],
                                                  max_lr=MAX_LR[tag_name],
                                                  epochs=EPOCHS)
        
        loss_fn = nn.BCEWithLogitsLoss()
        loss_tr = SmoothBCEwLogits(smoothing=0.001)

        
        best_loss = np.inf
        
        for epoch in range(EPOCHS):
            if fine_tune_scheduler is not None:
                fine_tune_scheduler.step(epoch, model)

            train_loss = train_fn(model, optimizer, scheduler, loss_tr, trainloader, device)
            valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, device)
            print(f"SEED: {seed}, FOLD: {fold}, {tag_name}, EPOCH: {epoch}, train_loss: {train_loss:.6f}, valid_loss: {valid_loss:.6f}")
            
            if valid_loss < best_loss:
                best_loss = valid_loss
                oof[valid.index] = valid_preds
                torch.save(model.state_dict(), f"{tag_name}_FOLD{fold}_.pth")

        print(f"Best loss for seed {seed}, fold {fold}: {best_loss}")

        return oof



    fine_tune_scheduler = FineTuneScheduler(SCORED_ONLY_EPOCHS)

    pretrained_model = Model(
        num_features=num_features,
        num_targets=num_targets_scored+num_targets_nscored,
    )
    pretrained_model.to(device)

    # Train on scored + nonscored targets
    train_model(pretrained_model, 'ALL_TARGETS')
    

    # Load the pretrained model with the best loss
    pretrained_model = Model(
        num_features=num_features,
        num_targets=num_targets_scored+num_targets_nscored,
    )
    pretrained_model.load_state_dict(torch.load(f"ALL_TARGETS_FOLD{fold}_.pth"))
    pretrained_model.to(device)

    # Copy model without the top layer
    final_model = fine_tune_scheduler.copy_without_top(pretrained_model, num_features, num_targets_scored+num_targets_nscored, num_targets_scored)

    # Fine-tune the model on scored targets only
    oof = train_model(final_model, 'SCORED_ONLY', fine_tune_scheduler)

    # Load the fine-tuned model with the best loss
    model = Model(
        num_features=num_features,
        num_targets=num_targets_scored,
    )
    model.load_state_dict(torch.load(f"SCORED_ONLY_FOLD{fold}_.pth"))
    model.to(device)

    #--------------------- PREDICTION---------------------
    x_test = test_df[feat_cols].values
    testdataset = TestDataset(x_test)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)
    
    predictions = np.zeros((test_df.shape[0], num_targets_scored))
    predictions = inference_fn(model, testloader, device)
    return oof, predictions


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

oof.to_csv("oof_v1.csv.gz",index=False,compression="gzip")
sub.to_csv("sub_v1.csv.gz",index=False,compression="gzip")
