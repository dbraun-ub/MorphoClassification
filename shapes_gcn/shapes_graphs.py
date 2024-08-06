import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric
from shapes_gcn import load_data
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import GATv2Conv, global_mean_pool


def drop_landmarks(data, indexes):
    """
    Drop data for given landmarks in the tensor.

    Parameters:
    - data (torch.Tensor): The tensor of shape (n, 2) representing n landmarkers in 2D.
    - indexes (list of int): The list of indexes of the landmarkers to be dropped.

    Returns:
    - torch.Tensor: The new tensor with the specified landmarkers removed.
    """
    # Create a mask with True values for rows to keep
    mask = torch.ones(data.shape[0], dtype=bool)
    mask[indexes] = False

    # Use the mask to select rows to keep
    new_data = data[mask]

    return new_data


def add_middle_landmark(data, index1, index2):
    """
    Add an extra landmarker in the middle point between two given landmarkers.

    Parameters:
    - data (torch.Tensor): The tensor of shape (n, 2) representing n landmarkers in 2D.
    - index1 (int): The index of the first landmarker.
    - index2 (int): The index of the second landmarker.

    Returns:
    - torch.Tensor: The new tensor with the additional landmarker.
    """
    # Ensure the indices are within bounds
    assert 0 <= index1 < data.shape[0], "index1 is out of bounds"
    assert 0 <= index2 < data.shape[0], "index2 is out of bounds"

    # Compute the middle point
    middle_point = (data[index1] + data[index2]) / 2

    # Append the new landmarker to the data
    new_data = torch.cat((data, middle_point.unsqueeze(0)), dim=0)

    return new_data


def create_extra_nodes(x, extra_nodes, config):
    for i in extra_nodes:
        if i == "SD07":
            x = add_middle_landmark(
                x, config["SD_indexes"]["SD05"], config["SD_indexes"]["SD08"]
            )
        if i == "SD01p":
            x = torch.cat(
                (
                    x,
                    torch.Tensor(
                        [
                            x[config["SD_indexes"]["SD11"], 0],
                            x[config["SD_indexes"]["SD01"], 1],
                        ]
                    ).unsqueeze(0),
                ),
                dim=0,
            )

        if i == "SD04p":
            x = torch.cat(
                (
                    x,
                    torch.Tensor(
                        [
                            x[config["SD_indexes"]["SD11"], 0],
                            x[config["SD_indexes"]["SD04"], 1],
                        ]
                    ).unsqueeze(0),
                ),
                dim=0,
            )

        if i == "SD07p":
            x = torch.cat(
                (
                    x,
                    torch.Tensor(
                        [
                            x[config["SD_indexes"]["SD11"], 0],
                            (
                                x[config["SD_indexes"]["SD05"]][1]
                                + x[config["SD_indexes"]["SD08"]][1]
                            )
                            / 2,
                        ]
                    ).unsqueeze(0),
                ),
                dim=0,
            )

        if i == "SD09p":
            x = torch.cat(
                (
                    x,
                    torch.Tensor(
                        [
                            x[config["SD_indexes"]["SD11"], 0],
                            x[config["SD_indexes"]["SD09"], 1],
                        ]
                    ).unsqueeze(0),
                ),
                dim=0,
            )

        if i == "SD10p":
            x = torch.cat(
                (
                    x,
                    torch.Tensor(
                        [
                            x[config["SD_indexes"]["SD11"], 0],
                            x[config["SD_indexes"]["SD10"], 1],
                        ]
                    ).unsqueeze(0),
                ),
                dim=0,
            )

    return x


def create_all_graphs(
    conf_mat: np.ndarray,
    df_params: pd.DataFrame,
    edges: str,
    model: str,
    config: dict,
    num_feature_labels: list,
    target_labels: list,
    extra_nodes=[],
    remove_nodes=[],
):

    x = torch.tensor(conf_mat, dtype=torch.float)

    # Add extra nodes
    if extra_nodes:
        x = create_extra_nodes(x, extra_nodes, config)

    # Remove nodes
    if remove_nodes:
        indexes_to_remove = [config["SD_indexes"][i] for i in remove_nodes]
        x = drop_landmarks(x, indexes_to_remove)

    # One-hot encode sex (categorical)
    sex_tensor = torch.tensor(
        [1, 0] if df_params["sex"] == "F" else [0, 1], dtype=torch.float
    )

    dic_tensors = {"sex": sex_tensor}

    for i in num_feature_labels:
        dic_tensors.update({i: torch.tensor([df_params[i]], dtype=torch.float)})

    list_tensors = []

    for i in dic_tensors:
        print(i)
        list_tensors.append(dic_tensors[i])

    global_features = torch.cat(list_tensors, dim=0)
    global_features = global_features.unsqueeze(0)

    edge_index = torch.tensor(
        [config[edges][model]["source_nodes"], config[edges][model]["destin_nodes"]],
        dtype=torch.long,
    )

    # Add targets as an extra dimension
    list_targets = []

    for i in target_labels:
        list_targets.append(df_params[i])

    target = torch.tensor([list_targets], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, y=target)

    data.global_features = global_features

    return data


def normalize(
    data: Data, params: dict, num_feature_labels: list, target_labels: list,
):

    # Normalize features
    # By default features 0 and 1 represent the One-hot encode sex feature
    # So we start normalizing features from k >= 2
    k = 2
    for i, j in enumerate(num_feature_labels):
        data.global_features[:, k + i] = (
            data.global_features[:, 2] - params[f"{j}_mean"]
        ) / params[f"{j}_std"]

    # Normalize height and weight targets
    for i, label in enumerate(target_labels):
        data.y[:, i] = (data.y[:, i] - params[f"{label}_mean"]) / params[f"{label}_std"]

    return data


class MyDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        raw_files,
        edge_view,
        edge_model,
        config,
        num_feature_labels,
        target_labels,
        extra_nodes,
        remove_nodes,
        transform=None,
        pre_transform=normalize,
        params=None,
    ):
        self.raw_files = raw_files
        self.edge_view = edge_view
        self.edge_model = edge_model
        self.config = config
        self.num_feature_labels = num_feature_labels
        self.target_labels = target_labels
        self.extra_nodes = extra_nodes
        self.remove_nodes = remove_nodes
        self.params_labels = num_feature_labels + target_labels
        self.params = params
        super(MyDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.age_mean = (
            None  # Initialize instance variables for normalization parameters
        )
        self.age_std = None
        self.height_mean = None
        self.height_std = None
        self.weight_mean = None
        self.weight_std = None

    @property
    def raw_file_names(self):
        return self.raw_files

    @property
    def processed_file_names(self):
        return ["data.pt"]  # Name of the processed file

    def download(self):
        pass  # No download in this example

    def process(self):
        data_list = []  # This will hold the Data objects

        # Fill data_list with the Data objects.

        # Load data from files
        confs, df_params = load_data(
            confs_path=self.raw_paths[0],
            demog_path=self.raw_paths[1],
            sd_params_path=self.raw_paths[2],
        )

        print(confs.shape)

        for i in range(confs.shape[2]):

            data_list.append(
                create_all_graphs(
                    confs[:, :, i],
                    df_params.iloc[i, :],
                    edges=self.edge_view,
                    model=self.edge_model,
                    config=self.config,
                    num_feature_labels=self.num_feature_labels,
                    target_labels=self.target_labels,
                    extra_nodes=self.extra_nodes,
                    remove_nodes=self.remove_nodes,
                )
            )

        if self.pre_transform is not None:
            print("Normalizing data...")

            if self.params is None:
                # Training set
                # Define mean and std values computed from the data
                describe = df_params.describe()

                self.params = {}

                for i in self.params_labels:

                    self.params.update(
                        {
                            f"{i}_mean": describe[i]["mean"],
                            f"{i}_std": describe[i]["std"],
                        }
                    )

            data_list = [
                self.pre_transform(
                    data, self.params, self.num_feature_labels, self.target_labels
                )
                for data in data_list
            ]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def create_graphs(
    conf_mat: np.array,
    group: np.array,
    edges: str,
    model: str,
    config: dict,
    encode_3views=True,
) -> torch_geometric.data.data.Data:

    x = torch.tensor(conf_mat, dtype=torch.float)

    # One-hot encode sex
    sex_tensor = torch.tensor([1, 0] if group[1] == "F" else [0, 1], dtype=torch.float)
    age_tensor = torch.tensor([group[2]], dtype=torch.float)

    if encode_3views:
        view_indicator = torch.tensor(config["view_encoding"][edges], dtype=torch.float)

        # Combine these into one tensor
        global_features = torch.cat([age_tensor, sex_tensor, view_indicator], dim=0)
    else:
        # Combine these into one tensor
        global_features = torch.cat([age_tensor, sex_tensor], dim=0)

    global_features = global_features.unsqueeze(0)

    edge_index = torch.tensor(
        [config[edges][model]["source_nodes"], config[edges][model]["destin_nodes"]],
        dtype=torch.long,
    )

    # Add target as an extra dimension
    height = group[3]
    weight = group[4]
    target = torch.tensor([[height, weight]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, y=target)

    # Set the sex attribute
    data.sex = sex_tensor
    data.age = age_tensor
    if encode_3views:
        data.view_indicator = view_indicator
    data.global_features = global_features

    return data


def normalize_(data, params):

    # Normalize age feature
    data.age = (data.age - params["age_mean"]) / params["age_std"]
    data.global_features[:, 0] = (
        data.global_features[:, 0] - params["age_mean"]
    ) / params["age_std"]

    # Normalize height and weight targets
    data.y[:, 0] = (data.y[:, 0] - params["height_mean"]) / params["height_std"]
    data.y[:, 1] = (data.y[:, 1] - params["weight_mean"]) / params["weight_std"]

    return data


class MyDataset_(InMemoryDataset):
    def __init__(
        self,
        root,
        raw_files,
        edge_view,
        edge_model,
        config,
        encode_3views=False,
        transform=None,
        pre_transform=normalize_,
        params=None,
    ):
        self.raw_files = raw_files
        self.edge_view = edge_view
        self.edge_model = edge_model
        self.config = config
        self.encode_3views = encode_3views
        self.params = params
        super(MyDataset_, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.age_mean = (
            None  # Initialize instance variables for normalization parameters
        )
        self.age_std = None
        self.height_mean = None
        self.height_std = None
        self.weight_mean = None
        self.weight_std = None

    @property
    def raw_file_names(self):
        return self.raw_files

    @property
    def processed_file_names(self):
        return ["data.pt"]  # Name of the processed file

    def download(self):
        pass  # No download in this example

    def process(self):
        data_list = []  # This will hold your Data objects

        # Fill data_list with your Data objects.
        # Here you would iterate over your pose configurations,
        # converting each one into a Data object and appending it to data_list.

        # Load data from files
        group_clean = np.load(self.raw_paths[0], allow_pickle=True)
        conf = np.load(self.raw_paths[1], allow_pickle=True)

        print(group_clean.shape)

        # for conf,view,rep in zip([conf_front,conf_back,conf_side],
        # ['FA','FP','SD'],self.edge_model):
        # for conf,view,rep in zip([conf_back],['FP'],self.edge_model):

        print(conf.shape)

        for i in range(conf.shape[2]):

            data_list.append(
                create_graphs(
                    conf[:, :, i],
                    group_clean[i, :],
                    edges=self.edge_view,
                    model=self.edge_model,
                    config=self.config,
                    encode_3views=self.encode_3views,
                )
            )

        if self.pre_transform is not None:
            print("Normalizing data...")

            if self.params is None:
                # Training set
                # Define your mean and std values computed from your data
                self.age_mean, self.age_std = (
                    group_clean[:, 2].mean(),
                    group_clean[:, 2].std(),
                )
                self.height_mean, self.height_std = (
                    group_clean[:, 3].mean(),
                    group_clean[:, 3].std(),
                )
                self.weight_mean, self.weight_std = (
                    group_clean[:, 4].mean(),
                    group_clean[:, 4].std(),
                )

                self.params = {
                    "age_mean": self.age_mean,
                    "age_std": self.age_std,
                    "height_mean": self.height_mean,
                    "height_std": self.height_std,
                    "weight_mean": self.weight_mean,
                    "weight_std": self.weight_std,
                }

            data_list = [self.pre_transform(data, self.params) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SingleViewGATv2(torch.nn.Module):
    def __init__(
        self, num_node_features, num_global_features, hidden_channels, num_classes
    ):
        super(SingleViewGATv2, self).__init__()

        self.attention_heads = 8

        self.gat1 = GATv2Conv(
            num_node_features, hidden_channels, heads=self.attention_heads, concat=True
        )
        self.gat2 = GATv2Conv(
            hidden_channels * self.attention_heads,
            hidden_channels,
            heads=1,
            concat=False,
        )

        # Linear layers to process the concatenated global features and node features
        self.lin1 = torch.nn.Linear(
            hidden_channels + num_global_features, hidden_channels
        )
        self.lin2 = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, batch, global_features = (
            data.x,
            data.edge_index,
            data.batch,
            data.global_features,
        )

        x = F.relu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gat2(x, edge_index)

        x = global_mean_pool(x, batch)  # Global pooling.
        x = torch.cat([x, global_features], dim=1)  # Concatenate with global features.

        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x
