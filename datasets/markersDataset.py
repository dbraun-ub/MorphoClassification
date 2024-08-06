import os
from PIL import Image
import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data, Dataset


class FrontViewMarkersDataset(Dataset):
    def __init__(self, node_coords, features_df, view, transform=None):#, num_classes=5):
        self.node_coords = node_coords
        self.features_df = features_df
        self.view = view
        self.transform = transform
        # self.num_classes = 5
    
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

        if self.view == 'FA':
            self.edges = [(0,1), (1,0), 
                (1,3), (3,1),
                (3,2), (2,3),
                (3,4), (4,3),
                (2,8), (8,2),
                (4,9), (9,4), 
                (3,5), (5,3),
                (5,6), (6,5),
                (5,7), (7,5),
                (6,10), (10,6),
                (7,11), (11,7), 
                (11,13), (13,11),
                (13,15), (15,13),
                (10,12), (12,10),
                (12,14), (14,12)]
        else:
            print('Not implemented.')

        # Transform gender to boolean values M == True, F == False
        features['gender'] = features['gender'] == 'M'
        

        # Get node features and target labels
        node_features = torch.tensor(self.node_coords[:,:,idx], dtype=torch.float)
        edge_index = torch.tensor(self.edges, dtype=torch.long).t().contiguous()
        global_features = torch.tensor(features[4:], dtype=torch.float)
        target = torch.tensor(label_to_index[5][features["morpho-1"]], dtype=torch.long)
        
        # Create the data object
        data = Data(x=node_features, edge_index=edge_index, y=target)
        data.global_features = global_features.unsqueeze(0)



        return data



