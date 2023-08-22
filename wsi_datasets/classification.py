from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle

from torch.utils.data import Dataset
import h5py

class WSI_Classification_Dataset(Dataset):
    def __init__(self, 
                 df, 
                 data_source, 
                 target_transform = None,
                 index_col = 'slide_id',
                 target_col = 'label', 
                 use_h5 = False,
                 label_map = None,
                 label_map_race = None,
                 study = None
                 ):
        """
        Args:
        """
        self.label_map = label_map
        self.label_map_race = label_map_race
        self.data_source = data_source
        self.index_col = index_col
        self.target_col = target_col
        self.target_transform = target_transform
        self.data = df
        self.data.fillna('N', inplace=True)
        self.use_h5 = use_h5
        self.study = study

        self._prep_instance_weights()

    def _prep_instance_weights(self):

        race_counts = dict(self.data["race"].value_counts())
        N = sum(dict(self.data["race"].value_counts()).values())
        weight_per_race = {}

        for race in race_counts:
            race_count = race_counts[race]
            weight = N / race_count
            weight_per_race[race] = weight

        self.weights = [0] * int(N)  

        for idx in range(N):   
            y = self.data.loc[idx, "race"]                 
            self.weights[idx] = weight_per_race[y]

        self.weights = torch.DoubleTensor(self.weights)


    def __len__(self):
        return len(self.data)

    def get_ids(self, ids):
        return self.data.loc[ids, self.index_col]

    def get_labels(self, ids):
        return self.data.loc[ids, self.target_col]
    
    def get_caseID(self, ids):
        return self.data.loc[ids, "case_id"]
    
    def get_race(self, ids):
        if self.label_map_race:
            return self.label_map_race[self.data.loc[ids, "race"]]
        else:
            None
    

    def __getitem__(self, idx):
        
        slide_id = self.get_ids(idx)
        label = self.get_labels(idx)
        case_id = self.get_caseID(idx)
        race = self.get_race(idx)
        # print(slide_id, label)

        if self.label_map is not None:
            label = self.label_map[label]
        if self.target_transform is not None:
            label = self.target_transform(label)
        try:
            if self.use_h5:
                feat_path = os.path.join(self.data_source, 'h5_files', slide_id + '.h5')
                with h5py.File(feat_path, 'r') as f:
                    features = torch.from_numpy(f['features'][:])
            else:
                feat_path = os.path.join(self.data_source, 'pt_files', slide_id + '.pt')
                features = torch.load(feat_path)
        except:
            features = torch.from_numpy(np.zeros([100,768]).astype(np.float32))
    
        
        return {'img': features, 'label': label, "case_id": case_id, "race": race}





        

