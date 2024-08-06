import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, global_mean_pool

# num_node_features = 2 # 2D landmarks (x,y)
# num_global_features = 3 # age (1, numerical), sex (2, categorical), view (3, categorical)

# # hyperparamenter
# hidden_channels = 64

# num_classes = 2 # y (height and weight)

# This GCN model consists of two graph convolutional layers,
# each followed by a ReLU activation function. After the GCN layers,
# the node features are globally pooled to a graph-level representation
# (using mean pooling here), and this is then concatenated with the global features.
# After that, there are two linear layers, with a dropout layer in between.

# In your case, num_node_features would be 2 (for the 2D coordinates of each landmark),
# num_global_features would be 6 (for age, sex, and view_indicator),
# hidden_channels is a hyperparameter representing the number of hidden channels,
# and num_classes would be 2 (for height and weight).

# Next, you'd want to split your dataset into training and test sets,
# define a suitable loss function (e.g., mean squared error loss, since you're doing regression),
# and write the training and evaluation loops. If you need help with any of these steps,
# feel free to ask!


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


class GCN(torch.nn.Module):
    def __init__(
        self, num_node_features, num_global_features, hidden_channels, num_classes
    ):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
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
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch)  # Global pooling.
        x = torch.cat([x, global_features], dim=1)  # Concatenate with global features.

        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x


class SingleViewGAT(torch.nn.Module):
    def __init__(
        self, num_node_features, num_global_features, hidden_channels, num_classes
    ):
        super(SingleViewGAT, self).__init__()

        self.attention_heads = 8

        self.gat1 = GATConv(
            num_node_features, hidden_channels, heads=self.attention_heads, concat=True
        )
        self.gat2 = GATConv(
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


class MultiViewGAT(torch.nn.Module):
    def __init__(
        self, num_node_features, num_global_features, hidden_channels, num_classes
    ):
        super(MultiViewGAT, self).__init__()

        self.attention_heads = 8

        # Three GAT modules, one for each view
        self.gat_anterior = GATConv(
            num_node_features, hidden_channels, heads=self.attention_heads, concat=True
        )
        self.gat_posterior = GATConv(
            num_node_features, hidden_channels, heads=self.attention_heads, concat=True
        )
        self.gat_lateral = GATConv(
            num_node_features, hidden_channels, heads=self.attention_heads, concat=True
        )

        # Linear layers for combined features
        combined_features = (
            3 * hidden_channels * self.attention_heads + num_global_features
        )
        self.lin1 = torch.nn.Linear(combined_features, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x_anterior, edge_index_anterior = data.anterior.x, data.anterior.edge_index
        x_posterior, edge_index_posterior = data.posterior.x, data.posterior.edge_index
        x_lateral, edge_index_lateral = data.lateral.x, data.lateral.edge_index

        # Assuming that the batch indices for each view are the same (they should be,
        # given they are from the same data points)
        batch = data.anterior.batch

        # Process each view using the respective GAT module
        x_anterior = self.gat_anterior(x_anterior, edge_index_anterior)
        x_posterior = self.gat_posterior(x_posterior, edge_index_posterior)
        x_lateral = self.gat_lateral(x_lateral, edge_index_lateral)

        # Global mean pool each processed graph to obtain graph-level embeddings
        x_anterior = global_mean_pool(x_anterior, batch)
        x_posterior = global_mean_pool(x_posterior, batch)
        x_lateral = global_mean_pool(x_lateral, batch)

        # Concatenate graph embeddings and global features
        x = torch.cat([x_anterior, x_posterior, x_lateral, data.global_features], dim=1)

        # Pass through linear layers for final prediction
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x
