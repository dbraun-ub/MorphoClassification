import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.utils import to_networkx

plt.style.use("ggplot")


def plot_graph(data):

    # Convert to NetworkX graph
    G = to_networkx(data)

    # Create a subplot
    fig, ax = plt.subplots(1, 1, figsize=(3, 8))

    # Draw the graph using the node positions from your data
    pos = {node: data.x[node].tolist() for node in G.nodes()}
    nx.draw(G, pos, ax=ax, with_labels=True)

    ax.invert_yaxis()

    # Show the plot
    plt.show()


def sigma_clip(data, sigma_lower=3.0, sigma_upper=3.0):
    mean = np.mean(data)
    std = np.std(data)
    mask = (data > mean - sigma_lower * std) & (data < mean + sigma_upper * std)
    return mask


def remove_outliers(
    group_clean,
    age_min=5,
    age_max=90,
    sigma_lower=2.3,
    sigma_upper=2.3,
    age_bins=[5, 10, 20, 30, 40, 50, 60, 70, 80, 90],
):

    age_col_idx = 2
    combined_mask = np.array([True] * group_clean.shape[0])

    # Iterate over age bins and apply the sigma clip function
    for start_age, end_age in zip(age_bins[:-1], age_bins[1:]):
        age_mask = (group_clean[:, age_col_idx] >= start_age) & (
            group_clean[:, age_col_idx] < end_age
        )
        subset = group_clean[age_mask]
        print(
            "{} to {} years old, {} assessments.".format(
                start_age, end_age, subset.shape[0]
            )
        )

        # If there's data in this age bin
        if len(subset):
            # Apply sigma clipping on the desired columns
            masks = [
                sigma_clip(
                    subset[:, i], sigma_lower=sigma_lower, sigma_upper=sigma_upper
                )
                for i in range(2, 6)  # Should not reject based on age, set range(3, 6)
            ]
            mask = np.all(masks, axis=0)

            # Update the combined mask
            combined_mask[age_mask] = mask

    mask_cut = (group_clean[:, age_col_idx] < age_max) & (
        group_clean[:, age_col_idx] > age_min
    )

    combined_mask = combined_mask & mask_cut
    # The combined_mask now can be used to mask the entire numpy_array
    # masked_array = group_clean[combined_mask]

    return combined_mask


def plot_demo_data(group_data):

    plt.figure(figsize=(12, 10))

    for i, label in zip(
        range(4), ["Age (year)", "Height (cm)", "Weight (kg)", "BMI (kg/m2)"]
    ):
        ax = plt.subplot(2, 2, i + 1)
        ax.hist(group_data[:, 2 + i])
        ax.set_ylabel("N")
        ax.set_xlabel(label)

    plt.subplots_adjust(wspace=0.3)
    plt.show()

    plt.scatter(group_data[:, 2], group_data[:, 5], alpha=0.03)
    plt.ylabel("BMI (kg/m2)")
    plt.xlabel("Age (year)")
    plt.xlim(0, 100)
    plt.show()


def my_train_test(extra_features_and_targets, confs):

    # Create an array of indices and split it
    indices = np.arange(extra_features_and_targets.shape[0])
    train_indices, test_indices = train_test_split(
        indices, test_size=0.05, random_state=42, shuffle=True
    )
    print("Train: {}".format(train_indices.shape[0]))
    print("Test: {}".format(test_indices.shape[0]))

    # Use the indices to split your data
    extra_features_and_targets_train = extra_features_and_targets[train_indices]
    extra_features_and_targets_test = extra_features_and_targets[test_indices]

    confs_train = confs[:, :, train_indices]
    confs_test = confs[:, :, test_indices]

    return (
        extra_features_and_targets_train,
        extra_features_and_targets_test,
        confs_train,
        confs_test,
    )
