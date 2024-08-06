import os
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pandas as pd
from torch_geometric.data import DataLoader

from options import options
import utils
import shapes_gcn as shapes
from datasets import FrontViewMarkersDataset
from preprocess_utils import plot_graph

def construct_graph(df, view, edges):
    # Step 1: Graph Construction
    x_cols = [col for col in df.columns if col.startswith(f'x{view}_')]
    y_cols = [col for col in df.columns if col.startswith(f'y{view}_')]
    
    # Number of points (features per sample)
    n_pts = len(x_cols)  # Assuming each xFA_ corresponds to a point
    
    # Convert columns to numpy arrays
    x_array = np.array([df[x_col].values for x_col in x_cols]).T  # Shape: N x n_pts
    y_array = np.array([df[y_col].values for y_col in y_cols]).T  # Shape: N x n_pts
    
    # Combine x and y arrays along the last dimension to form the 'points'
    # points will have the shape (N, n_pts, C)
    points = np.stack([x_array, y_array], axis=-1)  # Shape: N x n_pts x C
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Check if indices are within bounds
    if edge_index.max().item() >= n_pts:
        raise ValueError("Edge index out of bounds!")

    return points, edge_index

def train_gcn(opt):
    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable cudnn's auto-tuner for reproducible results

    device = utils.set_device(opt.device)
    print(f"Using device: {device}")

    ## Variables
    print('-'*20)
    print(opt)
    print('-'*20)

    ## Define your dataset (A définir)
    # list_datasets = {
    #     'singleView': shapes.FrontViewMarkersDataset, 
    # } 

    # Chargement des données morphologiques
    train_df = pd.concat([pd.read_csv(os.path.join(opt.split_path, opt.split, opt.split + f'_fold_{i}.csv')) for i in opt.train_folds], axis=0)
    val_df   = pd.concat([pd.read_csv(os.path.join(opt.split_path, opt.split, opt.split + f'_fold_{i}.csv')) for i in opt.val_folds], axis=0)

    # Chargement des marqueurs calibrés ainsi que leur id patient
    markers_FA = np.load(os.path.join('data', 'conf_mat_clean_full_balanced_FA_3057_0123456789.npy'))
    markers_FP = np.load(os.path.join('data', 'conf_mat_clean_full_balanced_FP_3057_0123456789.npy'))
    markers_SD = np.load(os.path.join('data', 'conf_mat_clean_full_balanced_SD_3057_0123456789.npy'))
    group_idx = np.load(os.path.join('data', 'group_clean_full_balanced_3057_0123456789.npy'), allow_pickle=True)

    # mask listant les éléments train et val des données markers
    train_mask = [idx in train_df['patient_id'].values for idx in group_idx[:,0]]
    val_mask = [idx in val_df['patient_id'].values for idx in group_idx[:,0]]

    train_df.set_index('patient_id', inplace=True)
    train_df = train_df.loc[group_idx[train_mask][:,0]]
    train_df.reset_index(inplace=True)

    val_df.set_index('patient_id', inplace=True)
    val_df = val_df.loc[group_idx[val_mask][:,0]]
    val_df.reset_index(inplace=True)


    train_dataset = FrontViewMarkersDataset(markers_FA[:,:,train_mask], train_df, 'FA')#, num_classes=5)

    plot_graph(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)

    # Only work with the single viwe FA
    model = shapes.SingleViewGATv2(
        num_node_features=2, # x,y
        num_global_features=5,
        hidden_channels=64,
        num_classes=opt.num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    scheduler = StepLR(optimizer, step_size=opt.scheduler_step_size, gamma=opt.scheduler_gamma)

    for epoch in range(opt.num_epochs):
        model.train()
        model.to(device)

        running_loss = 0
        total = 0
        correct = 0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data.y)
            _, predicted = torch.max(outputs.data, 1)
            loss.backward()
            optimizer.step()

            # for logging
            running_loss += loss.item()
            total += data.y.size(0)
            correct += (predicted == data.y).sum().item()

            # if batch_idx % 100 == 0:
            #     print(f'Epoch [{epoch+1}/{opt.num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item()}, Acc: {100 * (predicted == data.y).sum().item() / data.y.size(0)}')

        print(f'Epoch [{epoch+1}/{opt.num_epochs}], Loss: {running_loss / len(train_loader)}, Acc: {100 * correct / total}')

        # Step the scheduler
        scheduler.step()

    global_features = ['gender', 'age', 'height', 'weigth', 'IMC']
    num_global_features = len(global_features)


    




    # markers = {
    #     'FA': ['FA02', 'FA04', 'FA05', 'FA06', 'FA07', 'FA08', 'FA09', 'FA11', 'FA12', 'FA13', 'FA14', 'FA15', 'FA16', 'FA18', 'FA19', 'FA20'],
    #     'SD': ['SD01', 'SD02', 'SD03', 'SD04', 'SD05', 'SD08', 'SD09', 'SD10', 'SD11'],
    #     'FP': ['FP03', 'FP04', 'FP05', 'FP06', 'FP07', 'FP09', 'FP10', 'FP11', 'FP12', 'FP13', 'FP14', 'FP16', 'FP17', 'FP19']
    #     }

    # view = 'FA'

    # conf_mat = np.zeros([train_df.shape[0], len(markers[view]), 2])

    # for i, m in enumerate(markers[view]):
    #     conf_mat[:, i, 0] = train_df['x_pix_' + m].values
    #     conf_mat[:, i, 1] = train_df['y_pix_' + m].values

    # a = conf_mat[:5]
    # # shapes.gpa(conf_mat[10])
    # print(conf_mat.shape)
    # # shapes.show(*shapes.demo_gpa(a))

    # r,s,t,d = shapes.gpa(a, -1) # Rotation, scale, translation, Remanian distance
    # D = []
    # for i in range(len(a)):
    #     D.append(a[i].dot(r[i]) * s[i] + t[i]) # X dot R * S + T
    

    # D = np.concatenate([D])


    
    # plt.figure()
    # plt.subplot(131)
    # plt.scatter(x=a[:,:,0],y=a[:,:,1])
    # plt.subplot(132)
    # plt.scatter(x=D[:,:,0],y=D[:,:,1])
    # plt.subplot(133)
    # Dn = D / np.linalg.norm(D, 'fro')
    # plt.scatter(x=Dn[:,:,0],y=Dn[:,:,1])
    # plt.show()

    # Pour l'alignement des points de validation, on aligne sur la shape moyenne des données d'entrainement. Il faut donc sauvegarder cette information quelque part:
    # r, s, t, d = opa(a[0], a[1])
	# a[1] = a[1].dot(r) * s + t




if __name__ == '__main__':
    opt = options()
    train_gcn(opt)