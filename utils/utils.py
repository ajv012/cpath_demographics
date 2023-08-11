import pickle
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, sampler
import torch.optim as optim
import pdb
import math
import os
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_MIL(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	label = torch.LongTensor([item[1] for item in batch])

	return [img, label]

def collate_features(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	coords = np.vstack([item[1] for item in batch])
	return [img, coords]
    
def get_simple_loader(dataset, batch_size=1, num_workers=1):
    kwargs = {'num_workers': 4, 'pin_memory': False, 'num_workers': num_workers} if device.type == "cuda" else {}
    loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_MIL, **kwargs)
    return loader 

def get_optim(model, args):
    if args.opt == "adamW":
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    else:
        raise NotImplementedError
    return optimizer

def print_network(net):
    num_params = 0
    num_params_train = 0
    print(str(net))    
    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n
    
    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_dir, patience=20, min_stop_epoch=50, verbose=False, better='min'):
        """
        train_args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            min_stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.patience_counter = 0
        self.min_stop_epoch = min_stop_epoch
        self.better = better
        self.verbose = verbose
        self.best_score = None
        self.save_dir = save_dir
        
        if better == 'min':
            self.best_score = np.Inf
        else:
            self.best_score = -np.Inf
        self.early_stop = False
        self.counter = 0

    def is_new_score_better(self, score):
        if self.better == 'min':
            return score < self.best_score
        else:
            return score > self.best_score

    def __call__(self, epoch, score, save_ckpt_fn, save_ckpt_kwargs):
        is_better = self.is_new_score_better(score)
        if is_better:
            print(f'score improved ({self.best_score:.6f} --> {score:.6f}).  Saving model ...')
            self.save_checkpoint(save_ckpt_fn, save_ckpt_kwargs)
            self.counter = 0
            self.best_score = score
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.min_stop_epoch:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, save_ckpt_fn, save_ckpt_kwargs):
        '''Saves model when score improves.'''
        save_ckpt_fn(save_dir = self.save_dir, **save_ckpt_kwargs)

def save_checkpoint(config, epoch, model, score, save_dir, fname = None):
    save_state = {'model': model.state_dict(),
                  'score': score,
                  'epoch': epoch,
                  'config': config}
    
    if fname is None:
        save_path = os.path.join(save_dir, f'ckpt_epoch_{epoch}.pth')
    else:
        save_path = os.path.join(save_dir, fname)

    torch.save(save_state, save_path)