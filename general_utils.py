import pandas as pd
import os
import pickle
def get_df_regime_label(df_regime_discrete, label_dict):
    df_regime_labelled = df_regime_discrete.copy()
    for col, label_sub_dict in label_dict.items():
        value_counts = df_regime_discrete[col].value_counts()
        for value, label in label_sub_dict.items():
            new_col = df_regime_labelled[col].replace(value, f'{col}_{label}_{value_counts[value]}')
            df_regime_labelled[col] = new_col
    return df_regime_labelled



def save_object(obj, filename, directory):
    """
    Save an object to a file using pickle in the specified directory.

    Parameters:
        obj (object): The object to be saved.
        filename (str): The name of the file where the object will be saved.
        directory (str): The directory where the file will be saved.
    """
    full_path = os.path.join(directory, filename)
    with open(full_path, 'wb') as file:
        pickle.dump(obj, file)


def load_object(filename, directory):
    """
    Load an object from a pickle file in the specified directory.

    Parameters:
        filename (str): The name of the file from which the object will be loaded.
        directory (str): The directory where the file is located.
    
    Returns:
        object: The object loaded from the file.
    """
    directory = directory.replace('\\', '/')
    if directory[-1] != '/':
        directory  = directory + '/'
    full_path = os.path.join(directory, filename)
    with open(full_path, 'rb') as file:
        return pickle.load(file)

