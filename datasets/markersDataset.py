import os
from PIL import Image
import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data, Dataset
from sklearn.linear_model import LinearRegression
from torch_geometric.data.data import BaseData


class MarkersDataset(Dataset):
    def __init__(self, node_coords, features_df, transform=True, n_classes=5, global_features_mean=None, global_features_std=None, weight_index=False):
        self.node_coords = node_coords
        self.features_df = features_df
        self.transform = transform
        self.n_classes = n_classes
        self.global_features_mean = global_features_mean
        self.global_features_std = global_features_std
        self.weight_index = weight_index

        self.label_to_index = {
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

        # clean unused dataframe data
        if 'morpho1_gender' in self.features_df.columns:
            self.features_df = self.features_df.drop(columns=['morpho1_gender'])

        # replace IMC with corpulence index
        if weight_index:
            # corpulence index = kg / m^3
            self.features_df["IMC"] = self.features_df["weight_kg"] / (self.features_df["height_cm"]/100)**3

        if (self.global_features_mean is None) or (self.global_features_std is None):
            self.preprocess_global_features()
    
    def preprocess_global_features(self):
        self.global_features_mean = []
        self.global_features_std = []
        for i in range(4):
            self.global_features_mean.append(np.mean(self.features_df.iloc[:, i+5]))
            self.global_features_std.append(np.std(self.features_df.iloc[:, i+5]))
    
    def construct_data(self, features, nodes, edges):
        global_features = [features['gender'] == 'M']
        if self.transform:
            for i in range(4):
                global_features.append((features.iloc[i+5] - self.global_features_mean[i]) / self.global_features_std[i])
        else:
            for i in range(4):
                global_features.append(features.iloc[i+5])

        # Get node features and target labels
        node_features = torch.tensor(nodes, dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        global_features = torch.tensor(global_features, dtype=torch.float)
        target = torch.tensor(self.label_to_index[self.n_classes][features["morpho-1"]], dtype=torch.long)
        
        # Create the data object
        data = Data(x=node_features, edge_index=edge_index, y=target)
        data.global_features = global_features.unsqueeze(0)

        return data
    
    def __len__(self):
        return len(self.features_df)

    def __getitem__(self):
        return NotImplementedError("Subclass must implement __getitem__ method")



class FrontViewMarkersDataset(MarkersDataset):
    def __init__(self, node_coords, features_df, view, transform=True, n_classes=5, global_features_mean=None, global_features_std=None, weight_index=False):
        super().__init__(node_coords, features_df, transform, n_classes, global_features_mean, global_features_std, weight_index)
        self.views = view
        if 'all' in self.views:
            self.views = ['FA', 'SD', 'FP']

    def aggregate_nodes_edges(self, edges):
        nodes = None
        for view in self.views:
            if nodes is None:
                nodes = self.node_coords[view]
                np_edges = np.array(edges[view])
            else:
                num_nodes = len(nodes)
                nodes = np.concatenate([nodes, self.node_coords[view]])
                np_edges = np.concatenate([np_edges, np.array(edges[view]) + num_nodes])

        return nodes, np_edges
    
    def __getitem__(self, idx):
        features = self.features_df.loc[idx]

        nodes = None
        np_edges = None
        edges = {'FA':[
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
                ],
                'SD':[
                    (0,1), (1,0),
                    (0,3), (0,3),
                    (1,2), (2,1),
                    (2,3), (3,2),
                    (4,5), (5,4),
                    (4,6), (6,4),
                    (5,6), (6,5),
                    (6,7), (7,6),
                    (7,8), (8,7),
                    (3,4), (4,3),
                    (3,5), (5,3)
                ],
                'FP':[
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
                ]}
        
        nodes, np_edges = self.aggregate_nodes_edges(edges)

        data = self.construct_data(features, nodes[:,:,idx], np_edges)
        return data



class Markers3dDataset(MarkersDataset):
    def __init__(self, node_coords, features_df, view, transform=True, n_classes=5, global_features_mean=None, global_features_std=None, weight_index=False):
        super().__init__(node_coords, features_df, transform, n_classes, global_features_mean, global_features_std, weight_index)

    def process_3d_graph(self, idx):
        # Extract coordonates
        x_FA = self.node_coords['FA'][:,0, idx]
        y_FA = -self.node_coords['FA'][:,1, idx]
        x_FP = -self.node_coords['FP'][:,0, idx]
        y_FP = -self.node_coords['FP'][:,1, idx]
        z_SD = self.node_coords['SD'][:,0, idx]
        y_SD = -self.node_coords['SD'][:,1, idx]

        FA_pts = np.concatenate([x_FA.reshape(-1,1), y_FA.reshape(-1,1)], axis=1)
        FP_pts = np.concatenate([x_FP.reshape(-1,1), y_FP.reshape(-1,1)], axis=1)

        # Ajout de marqueurs géometrique
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
        
        # Alignement de FP_pts sur FA_pts
        idx_corresp_FA = [2,4,17,18]
        idx_corresp_FP = [2,1,15,16]

        P = np.array([FA_pts[i] for i in idx_corresp_FA])
        Q = np.array([FP_pts[i] for i in idx_corresp_FP])

        scale, R, t = calculate_affine_transformation_n_points(P,Q)

        FP_pts_aligned = scale * FP_pts @ R + t

        # Alignement de l'axe y de la vue SD sur la vue FA
        _, model = align_y_coordinates(np.concatenate([y_FA[[2,6]], FP_pts_aligned[[2,5,17],1]]), np.concatenate([y_SD[[3,5]], y_SD[[3,4,8]]]))
        y_SD_aligned = model.predict(y_SD.reshape(-1, 1))

        # Construction de points 3d
        pts_FA = np.concatenate([FA_pts, np.zeros((len(FA_pts),1))], axis=1)
        pts_SD = np.concatenate([np.zeros((len(y_SD),1)), y_SD_aligned.reshape(-1,1), z_SD.reshape(-1,1)], axis=1)
        pts_FP = np.concatenate([FP_pts_aligned, np.zeros((len(FP_pts_aligned),1))], axis=1)

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

        ## positionnement des points des genoux par conservation de la longueur du segment. On effectue une rotation du point.
        # plan (z,y)
        pts_FA[10,2], pts_FA[10,1] = projection_circulaire(
                                        np.array([pts_SD[6,2],pts_SD[6,1]]), 
                                        np.array([pts_SD[7,2],pts_SD[7,1]]), 
                                        np.array([pts_FA[10,2],pts_FA[10,1]]))
        pts_FA[11,2], pts_FA[11,1] = projection_circulaire(
                                        np.array([pts_SD[6,2],pts_SD[6,1]]), 
                                        np.array([pts_SD[7,2],pts_SD[7,1]]), 
                                        np.array([pts_FA[11,2],pts_FA[11,1]]))
        
        pts_FP[8,2], pts_FP[8,1] = projection_circulaire(
                                        np.array([pts_SD[6,2],pts_SD[6,1]]), 
                                        np.array([pts_SD[7,2],pts_SD[7,1]]), 
                                        np.array([pts_FP[8,2],pts_FP[8,1]]))
        pts_FP[9,2], pts_FP[9,1] = projection_circulaire(
                                        np.array([pts_SD[6,2],pts_SD[6,1]]), 
                                        np.array([pts_SD[7,2],pts_SD[7,1]]), 
                                        np.array([pts_FP[9,2],pts_FP[9,1]]))

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
        
        ## Extractions du nuage de points 3d
        self.nodes = np.concatenate([
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

        self.edges = [
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

    def __getitem__(self, idx):
        features = self.features_df.loc[idx]

        self.process_3d_graph(idx)

        data = self.construct_data(features, self.nodes, self.edges)

        return data


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