import os
import types

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.sampler import WeightedRandomSampler

from sklearn.impute import KNNImputer

# from skorch import NeuralNetRegressor

import numpy as np
from tqdm import tqdm

from loss import LossComputer

from transformers import AdamW, BertTokenizer
from datasets import load_dataset

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

class DRODataset(Dataset):
    def __init__(self, X, y, g):
        self.X = torch.Tensor(X)
        self.y = torch.tensor(y)
        self.group_to_name = {i:name for (i,name) in enumerate(set(g))}
        self.name_to_group = {name:i for (i,name) in enumerate(set(g))}
        self.g = torch.tensor([self.name_to_group[i] for i in g])
        self.n_groups = len(set(g))
        self._group_array = torch.LongTensor(self.g)
        self._group_counts = (torch.arange(self.n_groups).unsqueeze(1)==self._group_array).sum(1).float()

    def __len__(self):
        return len(self.g)

    def __getitem__(self, idx):
        X_row = self.X[idx].float()
        y_row = self.y[idx].float()
        g_row = self.g[idx].float()
        sample = {"X": X_row, "y": y_row, "g":g_row}
        return sample
  
    def group_counts(self):
        return torch.bincount(self.g)

    def get_loader(self, train=False, reweight_groups=False, **kwargs):
        if not train: # Validation or testing
            assert reweight_groups is None
            shuffle = False
            sampler = None
        elif not reweight_groups: # Training but not reweighting
            shuffle = True
            sampler = None
        else: # Training and reweighting
            # When the --robust flag is not set, reweighting changes the loss function
            # from the normal ERM (average loss over each training example)
            # to a reweighted ERM (weighted average where each (y,c) group has equal weight) .
            # When the --robust flag is set, reweighting does not change the loss function
            # since the minibatch is only used for mean gradient estimation for each group separately
            group_weights = len(self)/self._group_counts
            weights = group_weights[self._group_array]

            # Replacement needs to be set to True, otherwise we'll run out of minority samples
            sampler = WeightedRandomSampler(weights, len(self), replacement=True)
            shuffle = False

        loader = DataLoader(
          self,
          shuffle=shuffle,
          sampler=sampler,
          **kwargs)
        return loader
  
    def group_str(self, group_idx):
        self.group_to_name[group_idx]


def run_epoch(epoch, model, optimizer, loader, loss_computer, # logger, csv_logger, 
              args, is_training, show_progress=False, 
              log_every=50, scheduler=None):
    """
    scheduler is only used inside this function if model is bert.
    """

    if is_training:
        model.train()
    else:
        model.eval()

    prog_bar_loader = tqdm(loader)

    with torch.set_grad_enabled(is_training):
        for batch_idx, batch in enumerate(prog_bar_loader):
            # Unpack batch, feed through model, compute loss
            if args['d']=='dhs':
                x, y, g = batch['X'].to(f'cuda:{model.device_ids[0]}'), batch['y'].to(f'cuda:{model.device_ids[0]}'), batch['g'].to(f'cuda:{model.device_ids[0]}')
                outputs = model(x)
                loss_main = loss_computer.loss(outputs.view(-1), y, g, is_training)
        
            elif args['d']=='liar':
                batch = {k: v.to(f'cuda:{model.device_ids[0]}') for k, v in batch.items()}
                outputs = model(**batch)
                loss_main = loss_computer.loss(outputs['logits'], 
                                            batch['labels'], 
                                            batch['labels'], 
                                            is_training)
      
            # Backpropagate
            if is_training:
                optimizer.zero_grad()
                loss_main.backward()
                optimizer.step()

            if is_training and (batch_idx+1) % log_every==0:
                loss_computer.reset_stats()

            if (not is_training) or loss_computer.batch_count > 0:
                if is_training:
                    loss_computer.reset_stats()


def train(model, criterion, dataset, args, epoch_offset):    
    # model = model.to(device)

    # process generalization adjustment stuff
    adjustments = [float(c) for c in args['generalization_adjustment'].split(',')]
    assert len(adjustments) in (1, dataset['train_data'].n_groups)
    if len(adjustments)==1:
        adjustments = np.array(adjustments* dataset['train_data'].n_groups)
    else:
        adjustments = np.array(adjustments)

    # Define loss, optimizer, and LR scheduler
    train_loss_computer = LossComputer(
                            criterion,
                            is_robust=args['robust'],
                            dataset=dataset['train_data'],
                            alpha=args['alpha'],
                            gamma=args['gamma'],
                            adj=adjustments,
                            step_size=args['robust_step_size'],
                            normalize_loss=args['use_normalized_loss'],
                            btl=args['btl'],
                            min_var_weight=args['minimum_variational_weight'])
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args['lr'],
        # momentum=0.9,
        weight_decay=args['weight_decay'])
    if args['scheduler']:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'min',
            factor=0.1,
            patience=5,
            threshold=0.0001,
            min_lr=0,
            eps=1e-08)
    else:
        scheduler = None

    # Iterate through epochs
    best_val_acc = 0
    for epoch in range(epoch_offset, epoch_offset+args['n_epochs']):
      
        # Training Set
        # logger.write('\nEpoch [%d]:\n' % epoch)
        print(f'Epoch {epoch}')
        print(f'Training')
        run_epoch(
            epoch, model, optimizer, 
            dataset['train_loader'],
            train_loss_computer, 
            args,
            is_training=True,
            show_progress=args['show_progress'],
            log_every=args['log_every'],
            scheduler=scheduler)

        # Validation Set
        print(f'Validation')
        val_loss_computer = LossComputer(
            criterion,
            is_robust=args['robust'],
            dataset=dataset['val_data'],
            step_size=args['robust_step_size'],
            alpha=args['alpha'])
        run_epoch(
            epoch, model, optimizer, 
            dataset['val_loader'],
            val_loss_computer, # logger, val_csv_logger, 
            args,
            is_training=False)

        # Test set; don't print to avoid peeking
        if dataset['test_data'] is not None:
            test_loss_computer = LossComputer(
                criterion,
                is_robust=args['robust'],
                dataset=dataset['test_data'],
                step_size=args['robust_step_size'],
                alpha=args['alpha'])
            run_epoch(
                epoch, model, optimizer, 
                dataset['test_loader'],
                test_loss_computer, # None, test_csv_logger, 
                args,
                is_training=False)

        # Inspect learning rates
        if (epoch+1) % 1 == 0:
            for param_group in optimizer.param_groups:
                curr_lr = param_group['lr']
                print(f'Current lr: {curr_lr}')

        if args['scheduler'] and args['model'] != 'bert':
            if args['robust']:
                val_loss, _ = val_loss_computer.compute_robust_loss_greedy(val_loss_computer.avg_group_loss, val_loss_computer.avg_group_loss)
            else:
                val_loss = val_loss_computer.avg_actual_loss
            scheduler.step(val_loss) #scheduler step to update lr at the end of epoch

        if epoch % args['save_step'] == 0:
            if args['d'] in ['dhs']:
                torch.save(model, os.path.join(args['log_dir'], args['d'], '%d_model.pth' % epoch))
            else:
                model.module.save_pretrained(os.path.join(args['log_dir'], args['d'], '%d_model' % epoch))

        if args['save_last']:
            if args['d'] in ['dhs']:
                torch.save(model, os.path.join(args['log_dir'], args['d'], 'last_model.pth'))
            else:
                model.module.save_pretrained(os.path.join(args['log_dir'], args['d'], 'last_model'))

        if args['save_best']:
            if args['robust'] or args['reweight_groups']:
                curr_val_acc = min(val_loss_computer.avg_group_acc)
            else:
                curr_val_acc = val_loss_computer.avg_acc
            print(f'Current validation accuracy: {curr_val_acc}')
            if curr_val_acc > best_val_acc:
                best_val_acc = curr_val_acc
                if args['d'] in ['dhs']:
                    torch.save(model, os.path.join(args['log_dir'], args['d'], 'best_model'))
                else:
                    model.module.save_pretrained(os.path.join(args['log_dir'], args['d'], 'best_model'))
                print(f'Best model saved at epoch {epoch}')

        if args['automatic_adjustment']:
            gen_gap = val_loss_computer.avg_group_loss - train_loss_computer.exp_avg_loss
            adjustments = gen_gap * torch.sqrt(train_loss_computer.group_counts)
            train_loss_computer.adj = adjustments
            print('Adjustments updated\n')
            for group_idx in range(train_loss_computer.n_groups):
              pass
        #         logger.write(
        #             f'  {train_loss_computer.get_group_name(group_idx)}:\t'
        #             f'adj = {train_loss_computer.adj[group_idx]:.3f}\n')
        # logger.write('\n')
        # wandb.finish()
    
    return {'train_loss': train_loss_computer,
            'val_loss':   val_loss_computer,
            'test_loss':  test_loss_computer}
        
# Args
args = {'s':'confounder',
        't':None,
        'c':None,
        'resume':False,
        'minority_fraction':None,
        'imbalance_ratio':None,
        'fraction':1.0,
        'root_dir':None,
        'reweight_groups':False,
        'augment_data':False,
        'val_fraction':0.1,
        'alpha':0.2,
        'generalization_adjustment':"0.0",
        'automatic_adjustment':False, 
        'robust_step_size':0.01,
        'use_normalized_loss':False,
        'btl':False,
        'hinge':False,
        'train_from_scratch':False,
        'scheduler':True,
        'weight_decay':0.001,
        'gamma':0.1,
        'minimum_variational_weight':0,
        'seed':0,
        'show_progress':False,
        'log_dir':'logs/',
        'log_every':50,
        'save_step':10,
        'save_best':False,
        'save_last':True,
        'clip':0.05}


args['d'] = 'liar'

# Get data and splits
dataset = load_dataset('liar')
train_data, test_data, val_data = dataset['train'], dataset['test'], dataset['validation']

# Tokenize
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def encode_dict(d):
    return tokenizer(d['statement'], truncation=True, padding='max_length')

train_data = train_data.map(encode_dict, batched=True)
test_data  = test_data.map(encode_dict, batched=True)
val_data   = val_data.map(encode_dict, batched=True)

# Rename 'label' key to 'labels'
train_data = train_data.map(lambda examples: {'labels': examples['label']}, batched=True)
test_data  = test_data.map(lambda examples: {'labels': examples['label']}, batched=True)
val_data   = val_data.map(lambda examples: {'labels': examples['label']}, batched=True)

# Format the dataset
def format_LIAR_dataset(dataset):
    g = np.array([int(d['label']) for d in dataset])
    dataset.n_groups = len(set(g))
    dataset._group_array = torch.LongTensor(g)
    dataset._group_counts = (torch.arange(dataset.n_groups).unsqueeze(1)==dataset._group_array).sum(1).float()
    dataset.group_counts = dataset._group_counts
    return dataset

train_data = format_LIAR_dataset(train_data)
test_data  = format_LIAR_dataset(test_data)
val_data   = format_LIAR_dataset(val_data)

args['batch_size'] = 16

# Fix formatting for dataloader
weights = (len(train_data)/train_data._group_counts)[train_data._group_array]
sampler = WeightedRandomSampler(weights, len(train_data), replacement=True)
train_data.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
train_dataloader = torch.utils.data.DataLoader(train_data, 
                                               batch_size=args['batch_size'],
                                               shuffle=False,
                                               sampler=sampler)

test_data.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args['batch_size'])

val_data.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=args['batch_size'])

# Generate data
data = {}
data['train_data']   = train_data  # train_data
data['val_data']     = val_data    # val_data
data['test_data']    = test_data   # test_data
data['train_loader'] = train_dataloader # train_loader
data['val_loader']   = val_dataloader   # val_loader
data['test_loader']  = test_dataloader  # test_loader

# BERT
from transformers import BertConfig, BertForSequenceClassification

config_class = BertConfig
model_class = BertForSequenceClassification

config = config_class.from_pretrained(
    'bert-base-uncased',
    num_labels=6,
    finetuning_task='liar')
model = model_class.from_pretrained(
    'bert-base-uncased',# '/content/drive/MyDrive/S&DS 632/logs/last_model',
    from_tf=False,
    config=config)

model = torch.nn.DataParallel(model, device_ids=[0,1])
model.to(f'cuda:{model.device_ids[0]}')

criterion = torch.nn.CrossEntropyLoss(reduction='none')

args['model'] = 'bert'
args['n_epochs'] = 5
args['lr'] = 1e-5

args['robust'] = False

loss_dict = train(model, criterion, data, args, epoch_offset=0)

testloss = loss_dict['test_loss']
print("Test Loss")
print(testloss.avg_acc)
print(testloss.avg_group_acc)

valloss = loss_dict['val_loss']
print("Validation Loss")
print(valloss.avg_acc)
print(valloss.avg_group_acc)