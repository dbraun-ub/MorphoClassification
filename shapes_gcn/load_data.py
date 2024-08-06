import ast
import pickle

import numpy as np
import pandas as pd


def find_n_epoch(filename):

    # Split the string on the underscore first
    parts = filename.split("checkpoint_")

    # Now, parts is ['checkpoint', '690.pth']. We want the second part, so we take parts[1]

    # Now, split the second part on the dot
    subparts = parts[1].split(".")

    # Now, subparts is ['690', 'pth']. We want the numeral part, so we take
    # subparts[0] and convert it to integer
    num = int(subparts[0])

    return num


def load_dict(path):

    # Load the dictionary
    with open(path, "rb") as f:
        loaded_dict = pickle.load(f)

    return loaded_dict


def parse_value(value):
    value = value.strip()
    # Check if the value represents a list
    if value.startswith("[") and value.endswith("]"):
        # Use ast.literal_eval to safely evaluate the list
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    else:
        # For non-list values, use ast.literal_eval if possible
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value


def extract_commented_rows(filepath):
    comments = {}
    with open(filepath, "r") as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("#"):
                line = line[1:].strip()
                key, value = [x.strip() for x in line.split("=", 1)]
                value = parse_value(value)
                comments[key] = value
            else:
                break
    return comments


def read_sheet_column(sheet, column, end_row, start_row=1):
    """
    The function creates a list from a column in the
    Excel's worksheet
    """

    a = []
    for row in sheet[column][start_row:end_row]:
        a.append(row.value)

    return a


def load_data(
    confs_path, demog_path, sd_params_path,
):

    confs = np.load(confs_path)

    demog = np.load(demog_path, allow_pickle=True)
    print(demog.shape)
    sd_params = np.load(sd_params_path)
    print(sd_params.shape)

    sd_parms_cols = [
        "assessment_id",
        "SD_ANG1",
        "SD_ANG2",
        "SD_ANG3",
        "SD_ANG4",
        "SD_ANG5",
        "SD_ANG6",
        "SD_DIST1",
        "SD_DIST2",
        "SD_DIST3",
        "SD_DIST4",
        "SD_DIST5",
        "SD_YPGRAV",
        "BWeH",
        "BWeHT",
        "SD_LEVERH",
        "SD_LEVERHT",
        "SD_MOMH",
        "SD_MOMHT",
        "SD_JFH",
        "SD_JFHT",
    ]

    df_params = pd.concat(
        [
            pd.DataFrame(sd_params, columns=sd_parms_cols),
            pd.DataFrame(
                demog,
                columns=["assessment_id2", "sex", "age", "height", "weight", "bmi"],
            ),
        ],
        axis=1,
    )

    # df_params = df_params.drop('assessment_id2', axis=1)
    df_params.age = df_params.age.astype("float64")
    df_params.height = df_params.height.astype("float64")
    df_params.weight = df_params.weight.astype("float64")
    df_params.bmi = df_params.bmi.astype("float64")

    print(df_params.shape)

    return confs, df_params
