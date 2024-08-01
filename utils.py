import os
import pandas as pd
import torch


def merge_df_with_markers(df_morpho, split_path='splits/'):
    """
    merge annotated morphological dataframe with the one with markers.
    Initially, the marker dataframe has one line per marker, i.e. several lines pet patient_id. 
    We pivot the table to have one line per patient_id then merge the tables to keep the patient_id from morpho df.
    """

    filename = os.path.join(split_path, 'coordinates_Batch1.txt')
    df_marqueurs = pd.read_csv(filename)
    df2 = df_marqueurs[['patient_id', 'marker', 'x_pix', 'y_pix']]
    df2 = df2[~df2['marker'].str.startswith('SV')]
    df2 = df2[~df2['marker'].str.startswith('FAC')]
    df2 = df2[~df2['marker'].str.startswith('SDC')]
    df2 = df2[~df2['marker'].str.startswith('FPC')]

    # Drop duplicates based on the new column
    df2['patient_marker'] = df2['patient_id'].astype(str) + '_' + df2['marker']
    df2 = df2.drop_duplicates(subset='patient_marker', keep='first').reset_index(drop=True)

    # Drop the patient_marker column if it's no longer needed
    df2 = df2.drop(columns=['patient_marker'])

    # Pivot table around the patient_id index, to generate columns for each marker coordinates' x and y.
    df_pivot = df2.pivot(index='patient_id', columns='marker', values=['x_pix', 'y_pix'])
    df_pivot.columns = ['_'.join(col).strip() for col in df_pivot.columns.values]
    df_pivot.reset_index(inplace=True)

    # Merge dataframes on the patient_id to keep indexes from df_morpho
    df = pd.merge(df_morpho, df_pivot, on='patient_id', how='left')

    return df


class EarlyStopping:
    """
    early stopping function to stop the training if there is no progression on the loss function.
    Parameters:
        patience -> number of epochs without progress before stopping
        min_delta -> margin allowd between the val loss and the best loss to decide if there was progress.
            the lower the value, the strictier is the test for progress
    """
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def set_device(opt_device):
    # device = torch.device("cpu")
    if opt_device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("cuda is not available")
            device = torch.device("cpu")
    elif opt_device == "xpu":
        import intel_extension_for_pytorch as ipex
        if ipex.xpu.is_available():
            device = "xpu"
        else:
            opt_device = "cpu"
            print("xpu is not available")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    return device