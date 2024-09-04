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
from PIL import Image, ImageDraw
import argparse

from options import options, save_options, load_options
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
    results = []
    list_predicted = []
    list_target = []
    list_outputs = []
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
            results.append((predicted == targets).tolist())
            list_outputs.append(outputs.data)
            list_predicted.append(predicted.tolist())
            list_target.append(targets.tolist())

            # print(outputs.data.shape)

    # raw predictions for each class
    list_outputs = torch.cat(list_outputs, dim=0)

    # flatten the list
    results = [item for sublist in results for item in sublist]
    list_predicted = [item for sublist in list_predicted for item in sublist]
    list_target = [item for sublist in list_target for item in sublist]
    # print(len(results))
    # print(total)
    # print(results)
    
    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total

    return val_loss, val_accuracy, results, list_predicted, list_target, list_outputs

def get_loader(opt, folds, markers, global_features_mean=None, global_features_std=None, visualize=False, datasetClass=FrontViewMarkersDataset):
    df = pd.concat([pd.read_csv(os.path.join(opt.split_path, opt.split, opt.split + f'_fold_{i}.csv')) for i in folds], axis=0)
    markers_masked = copy.deepcopy(markers)

    group_idx = np.load(os.path.join('data', 'group_clean_full_balanced_3565_0123456789.npy'), allow_pickle=True)
    mask = [idx in df['patient_id'].values for idx in group_idx[:,0]]

    # On fait matcher les indexes de df à ceux de group_idx
    df.set_index('patient_id', inplace=True)
    df = df.loc[group_idx[mask][:,0]]
    df.reset_index(inplace=True)

    # Applya mask on maerkers to only keep values from the corresponding fold
    for view in markers_masked.keys():
        markers_masked[view] = markers_masked[view][:,:,mask]

    dataset = datasetClass(markers_masked, df, opt.view, n_classes=opt.num_classes, transform=opt.transform, global_features_mean=global_features_mean, global_features_std=global_features_std)
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

    if visualize:
        if dataset[0].x.shape[1] == 2: 
            plot_graph(dataset[0])
        else:
            plot_graph_3d(dataset[0])

    return loader, dataset.global_features_mean, dataset.global_features_std, df

def log_to_file(filename, message):
    with open(filename, 'a') as f:
        f.write(message + '\n')


def eval_gcn(opt, model_path):
    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable cudnn's auto-tuner for reproducible results

    device = utils.set_device(opt.device)
    print(f"Using device: {device}")



    # Chargement des marqueurs calibrés
    markers = {
        'FA': np.load(os.path.join('data', 'proc_rotated_clean_full_balanced_FA_3565_0123456789.npy')),
        'SD': np.load(os.path.join('data', 'proc_rotated_clean_full_balanced_SD_3565_0123456789.npy')),
        'FP': np.load(os.path.join('data', 'proc_rotated_clean_full_balanced_FP_3565_0123456789.npy'))
    }

    if opt.view[0] == '3d':
        train_loader, train_global_features_mean, train_global_features_std, _ = get_loader(opt, opt.train_folds, markers, datasetClass=Markers3dDataset, visualize=False)
        val_loader, _, _, _ = get_loader(opt, opt.val_folds, markers, global_features_mean=train_global_features_mean, global_features_std=train_global_features_std,datasetClass=Markers3dDataset, visualize=False)
        test_loader, _, _, df = get_loader(opt, opt.test_folds, markers, global_features_mean=train_global_features_mean, global_features_std=train_global_features_std,datasetClass=Markers3dDataset, visualize=False)

        model = shapes.SingleViewGATv2(
            num_node_features=3, # x,y,z
            num_global_features=5,
            hidden_channels=64,
            num_classes=opt.num_classes
        )
    else:
        train_loader, train_global_features_mean, train_global_features_std, _ = get_loader(opt, opt.train_folds, markers, datasetClass=FrontViewMarkersDataset, visualize=False)
        val_loader, _, _, _ = get_loader(opt, opt.val_folds, markers, global_features_mean=train_global_features_mean, global_features_std=train_global_features_std,datasetClass=FrontViewMarkersDataset, visualize=False)
        test_loader, _, _, df = get_loader(opt, opt.test_folds, markers, global_features_mean=train_global_features_mean, global_features_std=train_global_features_std,datasetClass=FrontViewMarkersDataset, visualize=False)

        model = shapes.SingleViewGATv2(
            num_node_features=2, # x,y
            num_global_features=5,
            hidden_channels=64,
            num_classes=opt.num_classes
        )

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()
    model = model.to(device)

 
    

    criterion = nn.CrossEntropyLoss()

    
    test_loss, test_accuracy, results, list_predicted, list_targets, list_outputs = validation_step(model, criterion, test_loader, device)

    print(f'test loss: {test_loss:.3f}, test acc: {test_accuracy:.3f}')

    return results, test_accuracy, list_predicted, list_targets, df, list_outputs


# def somatograph_coords(ecto, meso, endo):
#     ecto *= 6
#     meso *= 6
#     endo *= 6
#     x = ecto - endo
#     y = 2*meso - endo + ecto
#     return x, y


# def somatograph_coords(ecto, meso, endo):
#     """
#     Convert 3D coordinates to a 2D isometric projection.

#     Parameters:
#     - points_3d: A numpy array of shape (N, 3) where N is the number of 3D points.

#     Returns:
#     - A numpy array of shape (N, 2) containing the 2D projected coordinates.
#     """

#     points_3d = np.array([[ecto, meso, endo]])
#     # Define the rotation angles for the isometric view
#     # angle = np.arcsin(np.tan(np.deg2rad(30)))  # 30 degrees rotation for isometric view

#     # # Rotation matrices for isometric projection
#     # rotation_matrix_x = np.array([
#     #     [1, 0, 0],
#     #     [0, np.cos(angle), -np.sin(angle)],
#     #     [0, np.sin(angle), np.cos(angle)]
#     # ])
#     # angle = np.deg2rad(-45)
#     # rotation_matrix_y = np.array([
#     #     [np.cos(angle), 0, np.sin(angle)],
#     #     [0, 1, 0],
#     #     [-np.sin(angle), 0, np.cos(angle)]
#     # ])

#     # # Combined rotation matrix for isometric projection
#     # isometric_projection_matrix = rotation_matrix_x @ rotation_matrix_y

#     isometric_projection_matrix = 1/np.sqrt(6) * np.array([
#         [np.sqrt(2),-np.sqrt(2),np.sqrt(2)],
#         [1,2,1],
#         [np.sqrt(3), 0, -np.sqrt(3)],
#     ])

#     angle = np.deg2rad(180)
#     rotation_matrix = np.array([
#         [np.cos(angle), 0, np.sin(angle)],
#         [0, 1, 0],
#         [-np.sin(angle), 0, np.cos(angle)]
#     ])

#     isometric_projection_matrix = isometric_projection_matrix @ rotation_matrix

#     # Apply the isometric projection matrix to the 3D points
#     points_2d = points_3d @ isometric_projection_matrix.T

#     # Return the 2D coordinates (x, y)
#     return points_2d[0, 0],points_2d[0, 1]

# # def somatograph_coords(a, b, c):
# #     x = 0.5 * (2 * b + c) / (a + b + c)
# #     y = (np.sqrt(3) / 2) * c / (a + b + c)
# #     return x, y

if __name__ == '__main__':
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="graph neural network evaluation.")
    
    # Add arguments
    parser.add_argument('--models', type=str, default="best", help="models to test: best or all")
    parser.add_argument('--vis_inference', type=str, default="all", help="visualize inference: all, errors or none.")
    parser.add_argument('--threshold', type=int, default=80, help="Methods accuracy threshold: default 70.")
    parser.add_argument('--folds', type=str, default="default", help="Methods accuracy threshold: default or all.")
    
    # Parse the arguments
    args = parser.parse_args()

    model_names = []
    skipped = 0
    inference = None
    ecto = None
    meso = None
    endo = None
    if args.models == "all":
        for entry in os.listdir('runs'):
            full_path = os.path.join('runs', entry)
            
            # Check if the entry is a folder and starts with the specific string
            if os.path.isdir(full_path) and entry.startswith('SingleViewGATv2'):
                model_names.append(entry)

    # on force le modèle à évaluer
    if args.models == "best":
        model_names = ["SingleViewGATv2_SD-FP_32_0.001_200_True_3_full_balanced_3_classes",
                    "SingleViewGATv2_SD-FP_32_0.001_200_True_3_full_balanced_3_classes"]

    
    for model_name in model_names:
        model_path = os.path.join('runs', model_name)

        opt = load_options(os.path.join(model_path, 'options.json'))
        if args.folds == "all":
            opt.test_folds = [0,1,2,3,4,5,6,7,8,9]
        results, accuracy, list_predicted, list_targets, df, list_outputs = eval_gcn(opt, os.path.join(model_path, 'model_best_acc.pth'))

        if accuracy < args.threshold:
            skipped += 1
            continue

        # convert prediction to boolean lists per class
        if inference is None:
            inference = results
            ecto = [pred == 0 for pred in list_predicted]
            meso = [pred == 1 for pred in list_predicted]
            endo = [pred == 2 for pred in list_predicted]
        else:
            # inference = inference + np.array(results).int()
            for i in range(len(results)):
                inference[i] += results[i]
                ecto[i] += [pred == 0 for pred in list_predicted][i]
                meso[i] += [pred == 1 for pred in list_predicted][i]
                endo[i] += [pred == 2 for pred in list_predicted][i]

    

    ## get ids of miss predictions


    # compute correct predictions
    new_pred = torch.tensor([ecto, meso, endo])

    
    if args.vis_inference == "all":
        all_entries = torch.ones_like(new_pred)
        n_inferences = len(model_names) - skipped
        high_confidence = (new_pred > 0.8*n_inferences)
        high_conficence_bad_inference = high_confidence * all_entries
    else:
        true_positives = torch.tensor([list_targets]) == torch.argmax(new_pred, dim=0)
        False_negatives = torch.tensor([list_targets]) != torch.argmax(new_pred, dim=0)

        # identify bad predictions with high confidence
        n_inferences = len(model_names) - skipped
        high_confidence = (new_pred > 0.8*n_inferences)
        high_conficence_bad_inference = high_confidence * False_negatives

    idx_high_conficence_bad_inference = torch.nonzero(high_conficence_bad_inference, as_tuple=True)
    
    if args.vis_inference != "none":
        # Parcourt l'ensemble des mauvaises inférences à haut niveau de confiance (tous les modèles ont "mal" prédit la même chose)
        class_labels = ['ecto', 'meso', 'endo']
        sigmoid = nn.Sigmoid()
        for i in range(idx_high_conficence_bad_inference[0].shape[0]):
            morpho_class = idx_high_conficence_bad_inference[0][i].item()
            idx = idx_high_conficence_bad_inference[1][i].item()
            # (morpho_class, idx) = item
            data = df.loc[idx]
            print(data)
            if args.models == "best":
                plt.figure(figsize=(18, 6))
                i_sub = 6
            else:
                plt.figure(figsize=(12, 6))
                i_sub = 3

            pix_markers = {
                'FA': np.load(os.path.join('data', 'conf_mat_clean_full_balanced_FA_3565_0123456789.npy')),
                'SD': np.load(os.path.join('data', 'conf_mat_clean_full_balanced_SD_3565_0123456789.npy')),
                'FP': np.load(os.path.join('data', 'conf_mat_clean_full_balanced_FP_3565_0123456789.npy'))
            }
            group_idx = np.load(os.path.join('data', 'group_clean_full_balanced_3565_0123456789.npy'), allow_pickle=True)
            group_idx_indice = np.where(group_idx[:,0] == data["patient_id"])[0]


            image_path = os.path.join(r'C:\Users\Daniel\OneDrive - SOLUTIONS BIOTONIX INC\Data\MORPHO_Batch1', data["filename"])
            plt.subplot(1, i_sub, 1)
            img = Image.open(image_path + 'FA.jpg')
            draw = ImageDraw.Draw(img)
            for point in pix_markers['FA'][:,:,group_idx_indice]:
                draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill='red', outline='red')
            plt.imshow(img)
            plt.title("FA")
            plt.text(10, 30, f'Prediction: {class_labels[morpho_class]}', fontsize=12, color='white', 
                bbox=dict(facecolor='red', alpha=0.5))

            plt.text(10, 75, f'Ground Truth: {class_labels[list_targets[idx]]}', fontsize=12, color='white', 
                bbox=dict(facecolor='green', alpha=0.5))
            plt.axis('off')

            plt.subplot(1, i_sub, 2)
            img = Image.open(image_path + 'SD.jpg')
            draw = ImageDraw.Draw(img)
            for point in pix_markers['SD'][:,:,group_idx_indice]:
                draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill='red', outline='red')
            plt.imshow(img)
            plt.title("SD")
            plt.axis('off')
            plt.subplot(1, i_sub, 3)
            img = Image.open(image_path + 'FP.jpg')
            draw = ImageDraw.Draw(img)
            for point in pix_markers['FP'][:,:,group_idx_indice]:
                draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill='red', outline='red')
            plt.imshow(img)
            plt.title("FP")
            plt.axis('off')
            
            

            if args.models == "best":                
                plt.subplot(122)
                prob = sigmoid(list_outputs[idx])
                prob /= torch.sum(prob)
                colors = ['red'] * len(prob)
                colors[list_targets[idx]] = 'orange'
                plt.bar(class_labels, prob, color=colors)
                plt.title("Output prediction per class")
                plt.ylabel("normalized score")
                plt.legend(handles=[
                    plt.Line2D([0], [0], color='orange', lw=4, label='Ground truth'),
                ])
            plt.show()
    


    # plot organized results
    combined = zip(inference, ecto, meso, endo, list_targets)


    combined_sorted = sorted(combined, reverse=True)

    inference, ecto, meso, endo, list_targets = zip(*combined_sorted)

    plt.figure()
    plt.plot(np.array(inference) / (len(model_names) - skipped))
    plt.title("Classification success of the test set images")
    plt.xlabel('test set entries')
    plt.ylabel('% of methods correctly classifying entries')

    plt.figure()
    plt.plot(np.arange(len(inference)) / len(inference), np.array(inference) / (len(model_names) - skipped))
    plt.xlabel('test set entries (%)')
    plt.ylabel('% of methods correctly classifying entries')

    plt.figure()
    plt.plot(np.arange(len(inference)) / len(inference), ecto)
    plt.plot(np.arange(len(inference)) / len(inference), meso)
    plt.plot(np.arange(len(inference)) / len(inference), endo)
    plt.legend(['ecto', 'meso', 'endo'])
    plt.xlabel('test set entries (%)')
    plt.ylabel('% of methods correctly classifying entry')

    plt.figure()
    plt.subplot(311)
    plt.plot(np.arange(len(inference)) / len(inference), np.array(ecto) / (len(model_names) - skipped))
    plt.plot(np.arange(len(inference)) / len(inference), [targ == 0 for targ in list_targets])
    plt.legend(['pred', 'gt'])
    plt.title('ecto')
    plt.subplot(312)
    plt.plot(np.arange(len(inference)) / len(inference), np.array(meso) / (len(model_names) - skipped))
    plt.plot(np.arange(len(inference)) / len(inference), [targ == 1 for targ in list_targets])
    plt.legend(['pred', 'gt'])
    plt.ylabel('% of methods correctly classifying entry')
    plt.title('meso')
    plt.subplot(313)
    plt.plot(np.arange(len(inference)) / len(inference), np.array(endo) / (len(model_names) - skipped))
    plt.plot(np.arange(len(inference)) / len(inference), [targ == 2 for targ in list_targets])
    plt.legend(['pred', 'gt'])
    plt.title('endo')
    plt.xlabel('test set entries (%)')

    plt.figure()
    plt.stackplot(np.arange(len(inference)) / len(inference), ecto, meso, endo, labels=['ecto', 'meso', 'endo'], alpha=0.6)
    plt.legend(['ecto', 'meso', 'endo'])
    plt.title("Classification predicted per entry per method")
    plt.xlabel("test set entries")
    plt.ylabel("pred class distribution among methods")

    new_pred = torch.tensor([ecto, meso, endo])
    new_pred = torch.argmax(new_pred, dim=0)

    correct = 0
    for target, pred in zip(list_targets, new_pred):
        correct += (target == pred)
    print(correct / len(new_pred))


    # (list_targets == torch.argmax(new_pred)
# (predicted == targets).sum().item()
    plt.show()
    

    # param_grid = {
    #     'batch_size': [16,32,64,96,128],
    #     'learning_rate': [1e-3],
    #     'view': [['3d'], ['FA', 'SD', 'FP'], ['FA', 'SD'], ['FA', 'FP'], ['SD', 'FP'], ['FA'], ['SD'], ['FP']],
    #     'scheduler_step_size': [200],
    #     'num_epochs': [200],
    #     'transform': [True],
    #     'num_classes': [3],
    #     'split': ['full_balanced_3_classes'],
    # }



    # param_combinations = list(ParameterGrid(param_grid))

    # for i, param in enumerate(param_combinations):
    #     for key in param.keys():
    #         setattr(opt, key, param[key])

    #     opt.model_name = 'SingleViewGATv2'
    #     opt.earlyStopping_patience = opt.num_epochs // 10
    #     opt.earlyStopping_min_delta = 1e-8
    #     opt.drop_rate = 0
    #     opt.log_name = f"{opt.model_name}_{'-'.join(opt.view)}_{opt.batch_size}_{opt.learning_rate}_{opt.scheduler_step_size}_{opt.transform}_{opt.num_classes}_{opt.split}" 

    #     save_options(opt)
    
    #     train_gcn(opt)