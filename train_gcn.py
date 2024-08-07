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
import copy

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


def validation_step(model, criterion, val_loader, device):
    model.eval()
    model.to(device)
    # if device == torch.device("xpu"):
    #     import intel_extension_for_pytorch as ipex
    #     model = ipex.optimize(model, dtype=torch.float32)
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            outputs = model(data)
            targets = data.y
            if opt.linear_target:
                sigmoid = nn.Sigmoid()
                outputs = sigmoid(outputs).squeeze(1)
                loss = criterion(outputs, targets / (opt.num_classes - 1))
                predicted = torch.round(outputs * (opt.num_classes - 1))
            else:
                loss = criterion(outputs, targets)
                _, predicted = torch.max(outputs.data, 1)

            val_loss += loss.item()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total

    return val_loss, val_accuracy

def get_loader(opt, folds, markers, global_features_mean=None, global_features_std=None):
    df = pd.concat([pd.read_csv(os.path.join(opt.split_path, opt.split, opt.split + f'_fold_{i}.csv')) for i in folds], axis=0)
    # markers_copy = copy.deepcopy(markers)

    group_idx = np.load(os.path.join('data', 'group_clean_full_balanced_3057_0123456789.npy'), allow_pickle=True)
    mask = [idx in df['patient_id'].values for idx in group_idx[:,0]]

    df.set_index('patient_id', inplace=True)
    df = df.loc[group_idx[mask][:,0]]
    df.reset_index(inplace=True)

    # Applya mask on maerkers to only keep values from the corresponding fold
    # for view in markers_copy.keys():
    #     markers_copy[view] = markers_copy[view][:,:,mask]

    dataset = FrontViewMarkersDataset(markers[opt.view][:,:,mask], df, opt.view, n_classes=opt.num_classes, global_features_mean=global_features_mean, global_features_std=global_features_std)
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    return loader, dataset.global_features_mean, dataset.global_features_std



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

    # Chargement des marqueurs calibrés
    markers = {
        'FA': np.load(os.path.join('data', 'proc_rotated_clean_full_balanced_FA_3057_0123456789.npy')),
        'SD': np.load(os.path.join('data', 'proc_rotated_clean_full_balanced_SD_3057_0123456789.npy')),
        'FP': np.load(os.path.join('data', 'proc_rotated_clean_full_balanced_FP_3057_0123456789.npy'))
    }

    train_loader, train_global_features_mean, train_global_features_std = get_loader(opt, opt.train_folds, markers)
    val_loader, _, _ = get_loader(opt, opt.val_folds, markers, global_features_mean=train_global_features_mean, global_features_std=train_global_features_std)
    test_loader, _, _ = get_loader(opt, opt.test_folds, markers, global_features_mean=train_global_features_mean, global_features_std=train_global_features_std)

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
        for _, data in enumerate(train_loader):
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


        val_loss, val_accuracy = validation_step(model, criterion, val_loader, device)
        _, test_accuracy = validation_step(model, criterion, test_loader, device)

        print(f'Epoch [{epoch+1}/{opt.num_epochs}], train loss: {running_loss / len(train_loader):.3f}, train acc: {100 * correct / total:.3f}, val loss: {val_loss:.3f}, val acc: {val_accuracy:.3f}, test acc: {test_accuracy:.3f}')

        scheduler.step()



if __name__ == '__main__':
    opt = options()
    train_gcn(opt)