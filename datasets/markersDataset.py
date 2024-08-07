import os
from PIL import Image
import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data, Dataset


class FrontViewMarkersDataset(Dataset):
    def __init__(self, node_coords, features_df, view, transform=None, n_classes=5, global_features_mean=None, global_features_std=None):
        self.node_coords = node_coords
        self.features_df = features_df
        self.view = view
        self.transform = transform
        self.n_classes = n_classes
        self.global_features_mean = global_features_mean
        self.global_features_std = global_features_std
        
        if (self.global_features_mean is None) or (self.global_features_std is None):
            self.preprocess_global_features()
    
    def preprocess_global_features(self):
        self.global_features_mean = []
        self.global_features_std = []
        for i in range(4):
            self.global_features_mean.append(np.mean(self.features_df.iloc[:, i+5]))
            self.global_features_std.append(np.std(self.features_df.iloc[:, i+5]))


    def __len__(self):
        return len(self.features_df)
    
    def __getitem__(self, idx):
        features = self.features_df.loc[idx]


        label_to_index = {
            5:{
                'ecto': 0, 
                'ecto-meso': 1,
                'meso': 2,
                'meso-endo': 3,
                'endo': 4
            },
            3:{
                'ecto': 0, 
                'meso': 1,
                'endo': 2
            }
        }

        # if self.view == 'FA':
        #     self.edges = [
        #         (0,1), (1,0), 
        #         (1,3), (3,1),
        #         (2,3), (3,2),
        #         (2,6), (6,2),
        #         (2,8), (8,2),
        #         (3,4), (4,3), 
        #         (3,5), (5,3),
        #         (4,9), (9,4),
        #         (4,7), (7,4),
        #         (5,6), (6,5),
        #         (5,7), (7,5),
        #         (6,7), (7,6),
        #         (6,10), (10,6),
        #         (7,11), (11,7), 
        #         (11,13), (13,11),
        #         (13,15), (15,13),
        #         (10,12), (12,10),
        #         (12,14), (14,12)
        #     ]
        if self.view == 'FA':
            self.edges = [
                (0,1), (1,0), 
                (1,3), (3,1),
                (2,3), (3,2),
                (2,8), (8,2),
                (3,4), (4,3), 
                (3,5), (5,3),
                (4,9), (9,4),
                (5,6), (6,5),
                (5,7), (7,5),
                (6,10), (10,6),
                (7,11), (11,7), 
                (11,13), (13,11),
                (13,15), (15,13),
                (10,12), (12,10),
                (12,14), (14,12)
            ]
            # nodes = self.node_coords['FA']
        elif self.view == 'SD':
            self.edges = [
                (0,1), (1,0),
                (0,3), (0,3),
                (1,2), (2,1),
                (2,3), (3,2),
                (3,6), (6,3),
                (4,5), (5,4),
                (5,7), (7,5),
                (4,7), (7,4),
                (7,8), (8,7),
                (3,4), (4,3),
                (3,5), (5,3)
            ]
            # nodes = self.node_coords['SD']
        elif self.view == 'FP':
            self.edges = [
                (0,1), (1,0),
                (0,2), (2,0),
                (0,3), (3,0),
                (1,3), (3,1),
                (1,6), (6,1),
                (2,3), (3,2),
                (2,7), (7,2),
                (3,4), (4,3),
                (3,5), (5,3),
                (4,5), (5,4),
                (4,8), (8,4),
                (5,9), (9,5),
                (8,10), (10,8),
                (9,11), (11,9),
                (10,12), (12,10),
                (11,13), (13,11)
            ]
            # nodes = self.node_coords['FP']
        else:
            print('Not implemented.')

        # Transform gender to boolean values M == True, F == False
        features['gender'] = features['gender'] == 'M'


        for i in range(4):
            features.iloc[i+5] = (features.iloc[i+5] - self.global_features_mean[i]) / self.global_features_std[i]
        
        # Get node features and target labels
        node_features = torch.tensor(self.node_coords[:,:,idx], dtype=torch.float)
        edge_index = torch.tensor(self.edges, dtype=torch.long).t().contiguous()
        global_features = torch.tensor(features.iloc[4:9], dtype=torch.float)
        target = torch.tensor(label_to_index[self.n_classes][features["morpho-1"]], dtype=torch.long)
        
        # Create the data object
        data = Data(x=node_features, edge_index=edge_index, y=target)
        data.global_features = global_features.unsqueeze(0)

        return data



