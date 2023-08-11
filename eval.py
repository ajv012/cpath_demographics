from __future__ import print_function

import argparse
import pdb
import os
import math
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_transmil import TransMIL

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from wsi_datasets.classification import WSI_Classification_Dataset
from data.cls_default import tasks, label_dicts
from sklearn.metrics import balanced_accuracy_score

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
import json

from models.model_clam import CLAM_SB
from utils.core_utils import validate

torch.multiprocessing.set_sharing_strategy('file_system')

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def read_splits(split_dir, fold_idx = None):
    splits = {}
    for split in ['train', 'val', 'test']:
        if fold_idx is not None:
            split_path = os.path.join(split_dir, split + f'_{fold_idx}.csv')
        else:
            split_path = os.path.join(split_dir, split + '.csv')
        if os.path.isfile(split_path):
            df = pd.read_csv(split_path)
            splits[split] = df
    return splits

def build_datasets(splits, **kwargs):
    
    for k, df in splits.items():
        dataset = WSI_Classification_Dataset(df, **kwargs)
        dataloader = DataLoader(dataset, batch_size=1, shuffle = (k == 'train'), num_workers=2)
        splits[k] = dataloader
        print(f'split: {k}, n: {len(dataloader)}')
    return splits

def merge_dict(main_dict, new_dict):
    for k, v in new_dict.items():
        if k not in main_dict:
            main_dict[k] = []
        main_dict[k].append(v)
    return main_dict

def main(args):
    dataset_kwargs = dict(data_source = args.data_source, 
                         label_map = args.label_map,
                         target_col = args.target_col,
                         label_map_race = args.label_map_race,
                         )
    splits = read_splits(args.split_dir, fold_idx=None)
    print('successfully read splits for: ', list(splits.keys()))

    # reading all splits, but only want test
    loader =  build_datasets(splits, **dataset_kwargs)['test']    
    
    print('\nInit Model...', end=' ')
    model_dict = {"in_dim": args.in_dim,
                  'n_classes': args.n_classes}
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_mb']:
    
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict)
        else:
            raise NotImplementedError
    elif args.model_type == "mil":
        model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, "in_dim": args.in_dim}
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)       
    elif args.model_type == "transmil":
        model_dict = {"n_classes": args.n_classes, "in_dim": args.in_dim}
        model = TransMIL(**model_dict)

    model.cuda()   
    print_network(model)

    if os.path.isdir(args.ckpt_path):
        print('looking for ckpts in a directory!')
        ckpts = [file for file in os.listdir(args.ckpt_path) if os.path.splitext(file)[-1] == '.pth']
        ckpts = [os.path.join(args.ckpt_path, file) for file in ckpts]
    else:
        ckpts = [args.ckpt_path]

    tags = [os.path.splitext(os.path.basename(ckpt))[0] for ckpt in ckpts]
    all_test_results = {}
    for ckpt, tag in zip(ckpts, tags):
        print("Going for checkpoint {}".format(tag))
        state_dict = torch.load(ckpt, map_location='cpu')['model']
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print('missing keys: ', missing_keys)
        print('unexpected keys: ', unexpected_keys)
        test_results, test_dump = validate(model, loader, loss_fn = nn.CrossEntropyLoss(), 
                                    print_every = args.print_every, dump_results = True, eval_mode=True)
        print('finished eval ckpt: ', os.path.basename(ckpt))
        print(test_results)
        
        all_test_results = merge_dict(all_test_results, test_results)
        filename = os.path.join(args.results_dir, f'{tag}_results.pkl')
        save_pkl(filename, test_dump)

    final_dict = {}
    final_dict = {'tags': tags}
    final_dict.update({k + '_test':v for k,v in all_test_results.items()})

    with open(os.path.join(args.results_dir, 'summary.json'), 'w') as f:
        f.write(json.dumps(final_dict, sort_keys=True, indent=4))

    final_df = pd.DataFrame(final_dict)
    save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

# Generic eval settings
print("parsing arguments")
parser = argparse.ArgumentParser(description='Configurations for WSI Eval')
parser.add_argument('--results_dir', default='./eval_results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use')
parser.add_argument('--data_source', type=str, default=None, 
                    help='')
parser.add_argument('--ckpt_path', type=str, default=None, 
                    help='manually specify the path to model checkpoint')
parser.add_argument('--target_col', type=str, default='label')
parser.add_argument('--model_type', type=str, 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model')
parser.add_argument('--task', type=str)
parser.add_argument('--in_dim', default=1024, type=int, help='dim of input features')
parser.add_argument('--print_every', default=100, type=int, help='how often to print')


if __name__ == "__main__":

    args = parser.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.label_map = label_dicts[args.task]

    args.label_map_race = label_dicts["adv_train"]
    args.n_classes_race = len(set(list(args.label_map_race.values())))
    print('label map race: ', args.label_map_race)

    print('task: ', args.task)
    print('label map: ', args.label_map)
    args.n_classes = len(set(list(args.label_map.values())))
    
    args.split_dir = os.path.join('splits_20fold', args.split_dir)
    
    print('split_dir: ', args.split_dir)

    args.results_dir = os.path.join(args.results_dir, str(args.exp_code))
    os.makedirs(args.results_dir, exist_ok=True)

    print("\n################### Settings ###################")
    for key, val in vars(args).items():
        print("{}:  {}".format(key, val))       
    with open(os.path.join(args.results_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(vars(args), sort_keys=True, indent=4))
    
    results = main(args)
    print("finished!")
 
