import os

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from preprocess_utils import plot_demo_data, remove_outliers
from sklearn.model_selection import train_test_split


def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory {directory_path} created.")
    else:
        print(f"Directory {directory_path} already exists.")


"""
The module processes shapes in different views separately aligned using GPA and saved
as serialized numpy arrays.

1 - Assessments with outlier demographic (age, height, weight and BMI)
data are identified and removed from the dataset for training.

2 - Train and test split is made.
"""


@click.command()
@click.option(
    "--shapes_fa",
    default="data/proc_rotated_clean_full_balanced_FA_3057_0123456789.npy",
    help="Path to a numpy array containing GPA aligned configuration matrices - frontal view.",
)
@click.option(
    "--shapes_fp",
    default="data/proc_rotated_clean_full_balanced_FP_3057_0123456789.npy",
    help="Path to a numpy array containing GPA aligned configuration matrices - back view.",
)
@click.option(
    "--shapes_sd",
    default="data/proc_rotated_clean_full_balanced_SD_3057_0123456789.npy",
    help="Path to a numpy array containing GPA aligned configuration matrices - side view.",
)
@click.option(
    "--group",
    default="data/group_clean_full_balanced_3057_0123456789.npy",
    help="Path to a numpy array containing group data (demographic).",
)
@click.option(
    "--sd_params",
    default="data/coordinates_Batch1.txt",
    help="Path to a CSV file containing anthropometric parameters derived from the side view.",
)
@click.option(
    "--write_data", default=True, type=bool, help="Write files.",
)
@click.option(
    "--root_out", required=True, help="Root path for the output files.",
)
@click.option(
    "--views",
    default="all",
    type=str,
    help="Specify which views to process: 'FA', 'FP', 'SD', or 'all'.",
)
def run_train_preproc(
    shapes_fa: str,
    shapes_fp: str,
    shapes_sd: str,
    group: str,
    sd_params: str,
    write_data: bool,
    root_out: str,
    views: str,
):

    print("Loading dataset...")
    proc_rotated_clean_fa, proc_rotated_clean_fp, proc_rotated_clean_sd = (
        None,
        None,
        None,
    )

    if views.lower() in ["fa", "all"]:
        proc_rotated_clean_fa = np.load(shapes_fa)
        print(f"Frontal view shape: {proc_rotated_clean_fa.shape}")

    if views.lower() in ["fp", "all"]:
        proc_rotated_clean_fp = np.load(shapes_fp)
        print(f"Back view shape: {proc_rotated_clean_fp.shape}")

    if views.lower() in ["sd", "all"]:
        proc_rotated_clean_sd = np.load(shapes_sd)
        print(f"Side view shape: {proc_rotated_clean_sd.shape}")

    group_clean = np.load(group, allow_pickle=True)
    print(f"Group data shape: {group_clean.shape}")

    # Subset parameters data
    df_sd_params = pd.read_csv(sd_params)

    sd_params_filtered = df_sd_params[
        df_sd_params["patient_id"].isin(group_clean[:, 0])
    ].to_numpy()
    print(f"SD parameters shape: {sd_params_filtered.shape}")

    consistency = pd.Series(group_clean[:, 0].astype("int64")).equals(
        pd.Series(sd_params_filtered[:, 0]).astype("int64")
    )
    print(consistency)

    plt.scatter(
        pd.Series(group_clean[:, 0].astype("int64")),
        pd.Series(sd_params_filtered[:, 0]).astype("int64"),
    )

    # Draw a line from (xmin, ymin) to (xmax, ymax)
    plt.plot([0, 800000], [0, 800000], color="black", linestyle="--", linewidth=2)

    plt.show()

    # Add BMI to the group data
    bmi = group_clean[:, 4] / (group_clean[:, 3] / 100) ** 2
    bmi = bmi.reshape(-1, 1)
    group_clean = np.hstack((group_clean, bmi))

    plot_demo_data(group_clean)

    # Mask assessments with outlier demographic data
    combined_mask = remove_outliers(group_clean, sigma_lower=2.3, sigma_upper=2.3)
    plot_demo_data(group_clean[combined_mask])

    # Subsetting - drop assessments with outlier demographic data and parameters data
    demog_features = group_clean[combined_mask, :]
    sd_params_filtered = sd_params_filtered[combined_mask, :]

    # Initialize list for stacking configurations
    confs_list = []

    if proc_rotated_clean_fa is not None:
        front_confs = proc_rotated_clean_fa[:, :, combined_mask]
        confs_list.append(front_confs)

    if proc_rotated_clean_fp is not None:
        back_confs = proc_rotated_clean_fp[:, :, combined_mask]
        back_confs_mirrored = back_confs.copy()
        back_confs_mirrored[:, 0] = -back_confs[:, 0]
        confs_list.append(back_confs_mirrored)

    if proc_rotated_clean_sd is not None:
        side_confs = proc_rotated_clean_sd[:, :, combined_mask]
        confs_list.append(side_confs)

    # Stack all configurations if there are any
    if confs_list:
        confs = np.vstack(confs_list)
    else:
        raise ValueError("No views loaded based on the 'views' option provided.")

    # Print sample statistics
    for i, j in zip(range(2, 5), ["Age (years)", "Height (cm)", "Weight (kg)"]):
        print(
            f"{j}: M = {demog_features[:, i].mean().round(1)}, \
                SD = {demog_features[:, i].std().round(1)}"
        )

    # Split the data - train / test
    train_indices, test_indices = train_test_split(
        np.arange(demog_features.shape[0]),
        test_size=0.05,
        random_state=42,
        shuffle=True,
    )
    print(f"Train: {train_indices.shape[0]}")
    print(f"Test: {test_indices.shape[0]}")

    # Use the indices to split the data
    demog_features_train = demog_features[train_indices]
    demog_features_test = demog_features[test_indices]

    sd_params_train = sd_params_filtered[train_indices]
    sd_params_test = sd_params_filtered[test_indices]

    print(
        pd.Series(demog_features_train[:, 0].astype("int")).equals(
            pd.Series(sd_params_train[:, 0].astype("int"))
        )
    )
    plt.scatter(
        pd.Series(demog_features_train[:, 0].astype("int64")),
        pd.Series(sd_params_train[:, 0]).astype("int64"),
    )

    # Draw a line from (xmin, ymin) to (xmax, ymax)
    plt.plot([0, 800000], [0, 800000], color="black", linestyle="--", linewidth=2)

    plt.show()

    confs_train = confs[:, :, train_indices]
    confs_test = confs[:, :, test_indices]

    if write_data:
        ensure_directory_exists(root_out)
        ensure_directory_exists(root_out + "train/raw/")
        ensure_directory_exists(root_out + "test/raw/")
        np.save(root_out + "train/raw/demog_train.npy", demog_features_train)
        np.save(root_out + "test/raw/demog_test.npy", demog_features_test)
        np.save(root_out + "train/raw/sd_params_train.npy", sd_params_train)
        np.save(root_out + "test/raw/sd_params_test.npy", sd_params_test)
        np.save(root_out + "train/raw/confs_train.npy", confs_train)
        np.save(root_out + "test/raw/confs_test.npy", confs_test)


if __name__ == "__main__":
    run_train_preproc()
