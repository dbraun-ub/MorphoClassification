import os
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
import copy
from sklearn.model_selection import ParameterGrid
from torch.utils.tensorboard import SummaryWriter
import csv
from torch_geometric.utils import to_networkx

from options import options, save_options
import utils
import shapes_gcn as shapes
from datasets import FrontViewMarkersDataset, Markers3dDataset
from preprocess_utils import plot_graph

def plot_graph_3d(data):
    # Convert to NetworkX graph
    G = to_networkx(data)

    # Create a 3D subplot
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(122, projection='3d')

    # Extract positions from the data (assuming data.x has 3D coordinates)
    pos = {node: data.x[node].tolist() for node in G.nodes()}

    # Draw the graph
    for edge in G.edges():
        x_coords = [-pos[edge[0]][0], -pos[edge[1]][0]]
        y_coords = [pos[edge[0]][1], pos[edge[1]][1]]
        z_coords = [pos[edge[0]][2], pos[edge[1]][2]]
        ax.plot(x_coords, z_coords, y_coords, color='black')

    # Draw the nodes
    xs, ys, zs = zip(*[pos[node] for node in G.nodes()])
    ax.scatter(-np.array(xs), zs, ys, s=100)

    # Label the nodes
    for node in G.nodes():
        ax.text(-pos[node][0], pos[node][2], pos[node][1], str(node), size=10, zorder=1)

    # ax.invert_zaxis()  # Optional: invert z-axis if needed

    ax.set_aspect('equal')
    # Set the view orientation
    ax.view_init(elev=20, azim=110)

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
    if device == torch.device("xpu"):
        import intel_extension_for_pytorch as ipex
        model = ipex.optimize(model, dtype=torch.float32)
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

def get_loader(opt, folds, markers, global_features_mean=None, global_features_std=None, visualize=False, datasetClass=FrontViewMarkersDataset):
    df = pd.concat([pd.read_csv(os.path.join(opt.split_path, opt.split, opt.split + f'_fold_{i}.csv')) for i in folds], axis=0)
    markers_masked = copy.deepcopy(markers)

    group_idx = np.load(os.path.join('data', 'group_clean_full_balanced_3565_0123456789.npy'), allow_pickle=True)
    mask = [idx in df['patient_id'].values for idx in group_idx[:,0]]

    df.set_index('patient_id', inplace=True)
    df = df.loc[group_idx[mask][:,0]]
    df.reset_index(inplace=True)

    # Applya mask on maerkers to only keep values from the corresponding fold
    for view in markers_masked.keys():
        markers_masked[view] = markers_masked[view][:,:,mask]

    dataset = datasetClass(markers_masked, df, opt.view, n_classes=opt.num_classes, transform=opt.transform, global_features_mean=global_features_mean, global_features_std=global_features_std)
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    if visualize:
        if dataset[0].x.shape[1] == 2: 
            plot_graph(dataset[0])
        else:
            plot_graph_3d(dataset[0])

    return loader, dataset.global_features_mean, dataset.global_features_std

def log_to_file(filename, message):
    with open(filename, 'a') as f:
        f.write(message + '\n')


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
        'FA': np.load(os.path.join('data', 'proc_rotated_clean_full_balanced_FA_3565_0123456789.npy')),
        'SD': np.load(os.path.join('data', 'proc_rotated_clean_full_balanced_SD_3565_0123456789.npy')),
        'FP': np.load(os.path.join('data', 'proc_rotated_clean_full_balanced_FP_3565_0123456789.npy'))
    }

    if opt.view[0] == '3d':
        train_loader, train_global_features_mean, train_global_features_std = get_loader(opt, opt.train_folds, markers, datasetClass=Markers3dDataset, visualize=False)
        val_loader, _, _ = get_loader(opt, opt.val_folds, markers, global_features_mean=train_global_features_mean, global_features_std=train_global_features_std,datasetClass=Markers3dDataset, visualize=False)
        test_loader, _, _ = get_loader(opt, opt.test_folds, markers, global_features_mean=train_global_features_mean, global_features_std=train_global_features_std,datasetClass=Markers3dDataset, visualize=False)

        model = shapes.SingleViewGATv2(
            num_node_features=3, # x,y,z
            num_global_features=5,
            hidden_channels=64,
            num_classes=opt.num_classes
        ).to(device)
    else:
        train_loader, train_global_features_mean, train_global_features_std = get_loader(opt, opt.train_folds, markers, datasetClass=FrontViewMarkersDataset, visualize=False)
        val_loader, _, _ = get_loader(opt, opt.val_folds, markers, global_features_mean=train_global_features_mean, global_features_std=train_global_features_std,datasetClass=FrontViewMarkersDataset, visualize=False)
        test_loader, _, _ = get_loader(opt, opt.test_folds, markers, global_features_mean=train_global_features_mean, global_features_std=train_global_features_std,datasetClass=FrontViewMarkersDataset, visualize=False)

        model = shapes.SingleViewGATv2(
            num_node_features=2, # x,y
            num_global_features=5,
            hidden_channels=64,
            num_classes=opt.num_classes
        ).to(device)

 
    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    scheduler = StepLR(optimizer, step_size=opt.scheduler_step_size, gamma=opt.scheduler_gamma)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(opt.log_path, opt.log_name))

    # Initialize early stopping
    early_stopping = utils.EarlyStopping(patience=opt.earlyStopping_patience, min_delta=opt.earlyStopping_min_delta)

    ## Training loop
    best_val_acc = 0
    last_test_accuracy = 0
    best_epoch = 0
    list_val_accuracy = []
    for epoch in range(opt.num_epochs):
        model.train()
        if opt.device == "xpu":
            import intel_extension_for_pytorch as ipex
            model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.float32)
        
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

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)                  

        val_loss, val_accuracy = validation_step(model, criterion, val_loader, device)

        list_val_accuracy.append(val_accuracy)
        # mean_weights = np.exp(-np.array([4,3,2,1,0])**2 / 10) # Gaussian
        # mean_weights /= np.sum(mean_weights)
        # sliding_val_accuracy = list_val_accuracy[max(0, epoch-4):epoch+1]
        # mean_val_accuracy = np.mean(sliding_val_accuracy * mean_weights[-len(sliding_val_accuracy):])

        

        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        
        mean_val_accuracy = val_accuracy
        # mean_val_accuracy = (test_accuracy + val_accuracy) / 2

        if (mean_val_accuracy > best_val_acc) or ((mean_val_accuracy == best_val_acc) and (val_loss < best_val_loss)):
            best_val_acc = mean_val_accuracy
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f'runs/{opt.log_name}/model_best_acc.pth')

            test_loss, test_accuracy = validation_step(model, criterion, test_loader, device)

            last_test_loss = test_loss
            last_test_accuracy = test_accuracy

            # last_test_loss, last_test_accuracy = validation_step(model, criterion, test_loader, device)
            # last_test_accuracy = test_accuracy

            log_message = f'Epoch [{epoch+1}/{opt.num_epochs}], train loss: {avg_train_loss:.3f}, train acc: {train_accuracy:.3f}, val loss: {val_loss:.3f}, val acc: {val_accuracy:.3f}, test loss: {last_test_loss:.3f}, test acc: {last_test_accuracy:.3f}'
            print(log_message)
            log_to_file(f'runs/{opt.log_name}/log.txt', log_message)
        else:
            print(f'Epoch [{epoch+1}/{opt.num_epochs}], train loss: {avg_train_loss:.3f}, train acc: {train_accuracy:.3f}, val loss: {val_loss:.3f}, val acc: {val_accuracy:.3f}')

        scheduler.step()

        # Check for early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    evaluation_log = [
        opt.log_name,
        opt.model_name,             # Model
        best_val_loss,
        best_val_acc,    
        last_test_loss,
        last_test_accuracy,         # Accuracy
        '-'.join(opt.view),                   # View
        opt.dataset,                # dataset
        best_epoch,                 # Best epoch
        opt.val_folds[0],           # Val fold
        opt.test_folds[0],          # test fold
        opt.split,
        opt.learning_rate,
        opt.batch_size,
        opt.drop_rate,
        opt.scheduler_step_size,
        opt.scheduler_gamma,
        opt.num_epoch_unfreeze,
        opt.earlyStopping_patience,
        opt.earlyStopping_min_delta,
        opt.num_epochs,
        opt.transform,
        opt.num_classes,
        'gaus-val-loss'
    ]
    with open("runs/training_log.csv", 'a', newline='') as f:
        write = csv.writer(f)
        write.writerow(evaluation_log)



if __name__ == '__main__':
    opt = options()

    param_grid = {
        'batch_size': [16,32,64,96,128],
        'learning_rate': [1e-3],
        'view': [['3d'], ['FA', 'SD', 'FP'], ['FA', 'SD'], ['FA', 'FP'], ['SD', 'FP'], ['SD'], ['FP'], ['FA']],
        'scheduler_step_size': [200],
        'num_epochs': [200],
        'transform': [True],
        'num_classes': [3],
        'split': ['full_balanced_3_classes'],
        'val_folds': [[4],[5],[6],[7]],
        'test_folds': [[9]],
        'train_folds': [[0,1,2,3,5,6,7,8],[0,1,2,3,4,6,7,8],[0,1,2,3,4,5,7,8],[0,1,2,3,4,5,6,8]]
    }



    param_combinations = list(ParameterGrid(param_grid))

    for i, param in enumerate(param_combinations):
        skip_iteration = False
        for key in param.keys():
            setattr(opt, key, param[key])

        opt.model_name = 'SingleViewGATv2'
        opt.earlyStopping_patience = opt.num_epochs // 10
        opt.earlyStopping_min_delta = 1e-8
        opt.drop_rate = 0
        opt.log_name = f"{opt.model_name}_{'-'.join(opt.view)}_{opt.batch_size}_{opt.learning_rate}_{opt.scheduler_step_size}_{opt.transform}_{opt.num_classes}_{opt.split}_{opt.val_folds[0]}_{opt.test_folds[0]}" 

        if os.path.exists(os.path.join('runs', opt.log_name)):
            print(f'folder "{opt.log_name}" already exist.')
            continue

        for item in opt.val_folds:
            if (item in opt.train_folds) or (item in opt.test_folds):
                skip_iteration = True
                break

        for item in opt.test_folds:
            if (item in opt.train_folds) or (item in opt.val_folds):
                skip_iteration = True
                break

        if skip_iteration:
            print("skip iteration")
            continue

        save_options(opt)
    
        train_gcn(opt)