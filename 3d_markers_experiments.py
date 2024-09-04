import os
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from torch_geometric.data import Data
import torch
from torch_geometric.utils import to_networkx

from datasets import FrontViewMarkersDataset
from torch_geometric.loader import DataLoader
from preprocess_utils import plot_graph



def plot_graph_3d(data,image_path, idx):
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

    # Add images with markers
    data_path = r'C:\Users\Daniel\Workspace\shapes-gcn\exploration\raw'
    n = '3565'
    markers = {
            'FA': np.load(os.path.join(data_path, 'conf_mat_clean_full_balanced_FA_'+n+'_0123456789.npy')),
            'SD': np.load(os.path.join(data_path, 'conf_mat_clean_full_balanced_SD_'+n+'_0123456789.npy')),
            'FP': np.load(os.path.join(data_path, 'conf_mat_clean_full_balanced_FP_'+n+'_0123456789.npy'))
        }
    
    print(markers['FA'].shape)

    ax2 = fig.add_subplot(161)
    img = Image.open(image_path + 'FA.jpg')
    draw = ImageDraw.Draw(img)
    for point in markers['FA'][:,:,idx]:
        draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill='red', outline='red')
    ax2.imshow(img)
    ax2.axis('off')
    ax2 = fig.add_subplot(162)
    img = Image.open(image_path + 'SD.jpg')
    draw = ImageDraw.Draw(img)
    for point in markers['SD'][:,:,idx]:
        draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill='red', outline='red')
    ax2.imshow(img)
    ax2.axis('off')
    ax2 = fig.add_subplot(163)
    img = Image.open(image_path + 'FP.jpg')
    draw = ImageDraw.Draw(img)
    for point in markers['FP'][:,:,idx]:
        draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill='red', outline='red')
    ax2.imshow(img)
    ax2.axis('off')

    

def align_y_coordinates(y_front, y_side):
    """
    Aligner les coordonnées y de la vue de face avec celles de la vue latérale.
    
    Args:
    - y_front: Coordonnées y de la vue de face.
    - y_side: Coordonnées y de la vue latérale.
    
    Returns:
    - y_aligned: Coordonnées y réalignées correspondant à y_side.
    """
    # Régression linéaire pour trouver la transformation affine
    model = LinearRegression()
    model.fit(y_side.reshape(-1, 1), y_front)
    
    # Transformation des coordonnées y de la vue de face pour les aligner avec la vue latérale
    y_aligned = model.predict(y_side.reshape(-1, 1))
    
    return y_aligned, model

def calculate_affine_transformation_n_points(P,Q):
    """
    Calcule l'alignement de Q sur P en retournant l'échelle, la rotation, et la translation.
    
    Arguments:
    Q -- (NxD) Nuage de points source.
    P -- (NxD) Nuage de points cible.

    Retourne:
    scale -- Facteur d'échelle.
    R -- Matrice de rotation.
    t -- Vecteur de translation.
    """
    # Centrer les nuages de points
    centroid_Q = np.mean(Q, axis=0)
    centroid_P = np.mean(P, axis=0)
    Q_centered = Q - centroid_Q
    P_centered = P - centroid_P

    # Matrice de covariance
    H = Q_centered.T @ P_centered

    # Décomposition en valeurs singulières
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Gérer les cas de réflexion (si nécessaire)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Calculer le facteur d'échelle
    # scale = np.sum(S) / np.sum(Q_centered ** 2)
    scale = np.trace(P_centered.T @ Q_centered @ R) / np.sum(Q_centered ** 2)

    # Calculer la translation
    t = centroid_P - scale * centroid_Q @ R

    return scale, R, t

def apply_transformation(Q, scale, R, t):
    """
    Projette les points du nuage Q sur P en utilisant les paramètres d'alignement.
    
    Arguments:
    Q -- (NxD) Nuage de points source.
    scale -- Facteur d'échelle.
    R -- Matrice de rotation.
    t -- Vecteur de translation.
    
    Retourne:
    Q_aligned -- (NxD) Nuage de points aligné sur P.
    """
    # Appliquer l'échelle, la rotation, et la translation
    Q_aligned = scale * Q @ R + t
    return Q_aligned


def alignment_pipeline(P_idx, Q_idx, P_pts, Q_pts, legend=['Q', 'P->Q']):
    x_P = P_pts[:,0]
    y_P = P_pts[:,1]
    x_Q = Q_pts[:,0]
    y_Q = Q_pts[:,1]
    
    # Exemple de points de correspondance
    P = np.array([[x_P[i], y_P[i]] for i in P_idx])
    Q = np.array([[x_Q[i], y_Q[i]] for i in Q_idx])

    
    # Calcul de la transformation
    scale, R, t = calculate_affine_transformation_n_points(P, Q)

    # Appliquer la transformation
    transformed_points = apply_transformation(Q_pts, scale, R, t)
    
    if False:
        plt.figure(figsize=(15,6))
        plt.subplot(121)
        plt.scatter(P[:,0],P[:,1])
        plt.scatter(apply_transformation(Q, scale, R, t)[:,0],apply_transformation(Q, scale, R, t)[:,1])
        plt.legend(legend)
        plt.axis('equal')
        plt.title('points de correspondances')
        
        plt.subplot(122)
        plt.scatter(x_P,y_P)
        for i in range(len(x_P)):
            plt.annotate(str(i), (x_P[i], y_P[i]), textcoords="offset points", xytext=(5,-5), ha='center')
        # plt.scatter(x_FP,y_FP)
        plt.scatter(transformed_points[:,0],transformed_points[:,1])
        for i in range(len(transformed_points)):
            plt.annotate(str(i), (transformed_points[:,0][i], transformed_points[:,1][i]), textcoords="offset points", xytext=(5,-5), ha='center')
        plt.legend(legend)
        plt.axis('equal')
        plt.title('points alignés')

    return transformed_points

def plot_3d(pts_FA, pts_SD, pts_FP):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pts_FA[:, 0], pts_FA[:, 2], pts_FA[:, 1])
    ax.scatter(pts_SD[:, 0], pts_SD[:, 2], pts_SD[:, 1])
    ax.scatter(pts_FP[:, 0], pts_FP[:, 2], pts_FP[:, 1])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')

def projection_circulaire(A,B,C):
    # On cherche à trouver les nouvelles valeurs z de C et D.
    AB = B - A
    AB_norm = np.linalg.norm(AB)
    u_AB = AB / AB_norm
    # Calculer le vecteur AC
    AC = C - A
    # Projeter AC sur AB pour obtenir la composante sur le segment AB
    projection_length = np.dot(AC, u_AB)
    # Calculer le point C' sur AB
    C_prime = A + projection_length * u_AB

    return C_prime


def main(patient_index):
    ## Load 2d markers
    data_path = r'C:\Users\Daniel\Workspace\shapes-gcn\exploration\raw'
    split_path = 'splits'
    split = 'full_balanced'
    num_classes = 5
    folds = [0,1,2,3,4,5,6,7,8,9]
    views = ['FA', 'SD', 'FP']
    batch_size = 1
    # n = '3057'
    n = '3565'

    markers = {
            'FA': np.load(os.path.join(data_path, 'proc_rotated_clean_full_balanced_FA_'+n+'_0123456789.npy')),
            'SD': np.load(os.path.join(data_path, 'proc_rotated_clean_full_balanced_SD_'+n+'_0123456789.npy')),
            'FP': np.load(os.path.join(data_path, 'proc_rotated_clean_full_balanced_FP_'+n+'_0123456789.npy'))
        }

    df = pd.concat([pd.read_csv(os.path.join(split_path, split, split + f'_fold_{i}.csv')) for i in folds], axis=0)
    markers_masked = deepcopy(markers)

    group_idx = np.load(os.path.join(data_path, 'group_clean_full_balanced_'+n+'_0123456789.npy'), allow_pickle=True)
    print(group_idx[:5,:])
    mask = [idx in df['patient_id'].values for idx in group_idx[:,0]]

    print(df[df['patient_id'] == 14244])
    df.set_index('patient_id', inplace=True)
    df = df.loc[group_idx[mask][:,0]]
    df.reset_index(inplace=True)
    print(df.head())

    # Applya mask on maerkers to only keep values from the corresponding fold
    for view in markers_masked.keys():
        markers_masked[view] = markers_masked[view][:,:,mask]

    FA_dataset = FrontViewMarkersDataset(markers_masked, df, ['FA'], n_classes=num_classes)

    SD_dataset = FrontViewMarkersDataset(markers_masked, df, ['SD'], n_classes=num_classes)
    
    FP_dataset = FrontViewMarkersDataset(markers_masked, df, ['FP'], n_classes=num_classes)
    
    if False:
        ## Visualize markers
        plt.figure(figsize=(20,8))
        plt.subplot(131)

        x = FA_dataset[patient_index].x[:,0]
        y = -FA_dataset[patient_index].x[:,1]
        plt.scatter(x, y)
        for i in range(len(FA_dataset[patient_index].x)):
            plt.annotate(str(i), (x[i], y[i]), textcoords="offset points", xytext=(5,-5), ha='center')

        plt.axis('equal')
        plt.title('FA')

        plt.subplot(132)
        x = SD_dataset[patient_index].x[:,0]
        y = -SD_dataset[patient_index].x[:,1]
        plt.scatter(x, y)
        for i in range(len(SD_dataset[patient_index].x)):
            plt.annotate(str(i), (x[i], y[i]), textcoords="offset points", xytext=(5,-5), ha='center')
        plt.axis('equal')
        plt.title('SD')

        plt.subplot(133)
        x = FP_dataset[patient_index].x[:,0]
        y = -FP_dataset[patient_index].x[:,1]
        plt.scatter(x, y)
        for i in range(len(FP_dataset[patient_index].x)):
            plt.annotate(str(i), (x[i], y[i]), textcoords="offset points", xytext=(5,-5), ha='center')
        plt.axis('equal')
        plt.title('FP')


    ## Alignement de FP sur FA
    x_FA = FA_dataset[patient_index].x[:,0]
    y_FA = -FA_dataset[patient_index].x[:,1]
    x_FP = -FP_dataset[patient_index].x[:,0]
    y_FP = -FP_dataset[patient_index].x[:,1]

    FA_pts = np.concatenate([x_FA.view(-1,1), y_FA.view(-1,1)], axis=1)
    FP_pts = np.concatenate([x_FP.view(-1,1), y_FP.view(-1,1)], axis=1)

    FP_pts = np.vstack([FP_pts, 
                (FP_pts[4,:]+FP_pts[5,:])/2, 
                (FP_pts[8,:]+FP_pts[9,:])/2,
                (FP_pts[10,:]+FP_pts[11,:])/2,
                (FP_pts[12,:]+FP_pts[13,:])/2])

    FA_pts = np.vstack([FA_pts, 
                (FA_pts[6,:]+FA_pts[7,:])/2, 
                (FA_pts[10,:]+FA_pts[11,:])/2,
                (FA_pts[12,:]+FA_pts[13,:])/2],
                )

    FP_pts_aligned = alignment_pipeline(P_idx=[2,4,17,18], Q_idx=[2,1,15,16], P_pts=FA_pts, Q_pts=FP_pts, legend=['FA','FP->FA'])


    ## Alignement de SD sur FA
    z_SD = SD_dataset[patient_index].x[:,0]
    y_SD = -SD_dataset[patient_index].x[:,1]

    _, model = align_y_coordinates(np.concatenate([y_FA[[2,6]], FP_pts_aligned[[2,5,17],1]]), np.concatenate([y_SD[[3,5]], y_SD[[3,4,8]]]))

    y_SD_aligned = model.predict(y_SD.reshape(-1, 1))


    pts_FA = np.concatenate([FA_pts, np.zeros((len(FA_pts),1))], axis=1)
    pts_SD = np.concatenate([np.zeros((len(y_SD),1)), y_SD_aligned.reshape(-1,1), z_SD.view(-1,1)], axis=1)
    pts_FP = np.concatenate([FP_pts_aligned, np.zeros((len(FP_pts_aligned),1))], axis=1)


    # plot_3d(pts_FA, pts_SD, pts_FP)


    ## Projection 3d de points évidents
    # FA
    pts_FA[2,2] = pts_SD[3,2]
    pts_FA[4,2] = pts_SD[3,2] # extrapolation
    pts_FA[6,2] = pts_SD[5,2]
    pts_FA[7,2] = pts_SD[5,2] # extrapolation
    pts_FA[16,2] = pts_SD[5,2] # extrapolation

    # SD
    pts_SD[3,0] = pts_FA[2,0]
    pts_SD[4,0] = pts_FP[4,0]
    pts_SD[5,0] = pts_FA[6,0]

    # FP
    pts_FP[1,2] = pts_SD[3,2]
    pts_FP[2,2] = pts_SD[3,2] # extrapolation
    pts_FP[4,2] = pts_SD[4,2] # extrapolation
    pts_FP[5,2] = pts_SD[4,2]
    pts_FP[14,2] = pts_SD[4,2] # extrapolation
    
    # plot_3d(pts_FA, pts_SD, pts_FP)

    ## positionnement des points des genoux par conservation de la longueur du segment. On effectue une rotation du point.
    # plan (z,y)
    A = np.array([pts_SD[6,2],pts_SD[6,1]]) 
    B = np.array([pts_SD[7,2],pts_SD[7,1]])

    pts_FA[10,2], pts_FA[10,1] = projection_circulaire(A, B, np.array([pts_FA[10,2],pts_FA[10,1]]))
    pts_FA[11,2], pts_FA[11,1] = projection_circulaire(A, B, np.array([pts_FA[11,2],pts_FA[11,1]]))
    
    pts_FP[8,2], pts_FP[8,1] = projection_circulaire(A, B, np.array([pts_FP[8,2],pts_FP[8,1]]))
    pts_FP[9,2], pts_FP[9,1] = projection_circulaire(A, B, np.array([pts_FP[9,2],pts_FP[9,1]]))

    ## positionnement des chevilles
    pts_FP[12,2], pts_FP[12,1] = projection_circulaire(
                                    np.array([pts_FP[8,2],pts_FP[8,1]]), 
                                    np.array([pts_SD[8,2],pts_SD[8,1]]), 
                                    np.array([pts_FP[12,2],pts_FP[12,1]]))
    pts_FP[13,2], pts_FP[13,1] = projection_circulaire(
                                    np.array([pts_FP[9,2],pts_FP[9,1]]), 
                                    np.array([pts_SD[8,2],pts_SD[8,1]]), 
                                    np.array([pts_FP[13,2],pts_FP[13,1]]))
    
    ## Inclinaison de la tête.
    # La tête bouge beaucoup d'une vue à l'autre. Il faut donc choisir l'un ou l'autre. On fait le choix de prendre la vue de coté avec l'inclinaison de la vue de face.
    
    pts_SD[0,0], pts_SD[0,1] = projection_circulaire(
                                    np.array([pts_FA[0,0],pts_FA[0,1]]), 
                                    np.array([pts_FA[1,0],pts_FA[1,1]]), 
                                    np.array([pts_SD[0,0],pts_SD[0,1]]))
    
    pts_SD[1,0], pts_SD[1,1] = projection_circulaire(
                                    np.array([pts_FA[0,0],pts_FA[0,1]]), 
                                    np.array([pts_FA[1,0],pts_FA[1,1]]), 
                                    np.array([pts_SD[1,0],pts_SD[1,1]]))
    
    pts_SD[2,0], pts_SD[2,1] = projection_circulaire(
                                    np.array([pts_FA[0,0],pts_FA[0,1]]), 
                                    np.array([pts_FA[1,0],pts_FA[1,1]]), 
                                    np.array([pts_SD[2,0],pts_SD[2,1]]))
    
    ## Positionnement de la nuque
    # pts_FP[0,2], pts_FP[0,1] = projection_circulaire(
    #                                 np.array([pts_FA[2,2],pts_FA[2,1]]), 
    #                                 np.array([pts_SD[0,2],pts_SD[0,1]]), 
    #                                 np.array([pts_FP[0,2],pts_FP[0,1]]))
    # Position verticale uniquement
    pts_FP[0,2] = pts_FA[2,2]
    
    ## Alignement hypothetique des bras le long du corps
    # Alignement sur l'axe ventre - talon
    pts_FA[8,2], pts_FA[8,1] = projection_circulaire(
                                    np.array([pts_SD[5,2],pts_SD[5,1]]), 
                                    np.array([pts_SD[8,2],pts_SD[8,1]]), 
                                    np.array([pts_FA[8,2],pts_FA[8,1]]))
    pts_FA[9,2], pts_FA[9,1] = projection_circulaire(
                                    np.array([pts_SD[5,2],pts_SD[5,1]]), 
                                    np.array([pts_SD[8,2],pts_SD[8,1]]), 
                                    np.array([pts_FA[9,2],pts_FA[9,1]]))
    pts_FP[6,2], pts_FP[6,1] = projection_circulaire(
                                    np.array([pts_SD[5,2],pts_SD[5,1]]), 
                                    np.array([pts_SD[8,2],pts_SD[8,1]]), 
                                    np.array([pts_FP[6,2],pts_FP[6,1]]))
    pts_FP[7,2], pts_FP[7,1] = projection_circulaire(
                                    np.array([pts_SD[5,2],pts_SD[5,1]]), 
                                    np.array([pts_SD[8,2],pts_SD[8,1]]), 
                                    np.array([pts_FP[7,2],pts_FP[7,1]]))
    

    if False:
        plot_3d(pts_FA, pts_SD, pts_FP)

    ## Extractions du nuage de points 3d
    pts = np.concatenate([
        pts_SD[1:3], # tête
        pts_FP[0].reshape(1,3), # nuque
        ((pts_FA[2] + pts_FP[2]) / 2).reshape(1,3), # épaule droite
        ((pts_FA[4] + pts_FP[1]) / 2).reshape(1,3), # épaule gauche
        pts_FP[4:6], # dos
        pts_FA[6:8], # ventre
        ((pts_FA[8] + pts_FP[7]) / 2).reshape(1,3), # bras droit
        ((pts_FA[9] + pts_FP[6]) / 2).reshape(1,3), # bras gauche
        ((pts_FA[10] + pts_FP[9]) / 2).reshape(1,3), # genoux droit
        ((pts_FA[11] + pts_FP[8]) / 2).reshape(1,3), # genoux gauche
        pts_FP[12:14], # chevilles 
    ])
    print(pts.shape)

    if False:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1])
        
        for i in range(len(pts)):
            ax.text(pts[:,0][i], pts[:,2][i], pts[:,1][i], '%d' % i, size=12, zorder=1, color='k')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_aspect('equal')

    edges = [
        (0,1),(1,0),
        (0,2),(2,0),
        (1,2),(2,1),
        (2,3),(3,2),
        (2,4),(4,2),
        (3,4),(4,3),
        (3,6),(6,3),
        (3,7),(7,3),
        (3,9),(9,3),
        (4,5),(5,4),
        (4,8),(8,4),
        (4,10),(10,4),
        (5,6),(6,5),
        (5,8),(8,5),
        (5,12),(12,5),
        (6,7),(7,6),
        (6,11),(11,6),
        (7,8),(8,7),
        (7,11),(11,7),
        (8,12),(12,8),
        (11,14),(14,11),
        (12,13),(13,12)
    ]
    # Graphe
    
    data = Data(x=torch.tensor(pts, dtype=torch.float), edge_index=torch.tensor(np.array(edges), dtype=torch.long).t().contiguous())
    print(group_idx.shape)
    print(group_idx[13])
    print(group_idx[patient_index])
    print(patient_index)
    if (df['patient_id'] == group_idx[patient_index,0]) is False:
        print(f'patient_id {group_idx[patient_index,0]} not found')
    else:
        filename = df['filename'][df['patient_id'] == group_idx[patient_index,0]].values[0]
        # filename = df['filename'][df['patient_id'] == FA_dataset.features_df.loc[patient_index]].values[0]
        print('filename')
        print(filename)
        image_path = os.path.join(r'C:\Users\Daniel\OneDrive - SOLUTIONS BIOTONIX INC\Data\MORPHO_Batch1', filename)
        
        # print(image_path)
        plot_graph_3d(data,image_path, patient_index)

        plt.show()

import matplotlib.pyplot as plt



    


if __name__ == '__main__':

    for i in range(100):
        main(i)

