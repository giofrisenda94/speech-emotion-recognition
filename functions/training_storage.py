import pandas as pd
import numpy as np
from functions.path_functions import find_parent_filepath
import os.path



def store_df_np(data, filename, type):
    """
    Saves the X_train, X_test, y_train, or y_test pandas dataframe or numpy array into
    the train storage location for later use.  Pass the data, filename, and type,
    which is either a value of numpy or pandas
    """

    if type == "pandas":
        #Set Pathway
        parent = find_parent_filepath()
        path_to_pkl = parent + "/train_storage/"
        final_path = path_to_pkl + filename + ".pkl"

        #Save DataFrame to Pickle File
        data.to_pickle(final_path)

        #Check to See if File Exists
        if os.path.isfile(final_path) == True:
            return "File Saved to Pickle"
        elif os.path.isfile(final_path) == False:
            return "File Not Saved, Try Again"


    elif type == "numpy":
        #Set Pathway
        parent = find_parent_filepath()
        path_to_pkl = parent + "/train_storage/"
        final_path = path_to_pkl + filename + ".npy"

        #Save DataFrame to Pickle File
        np.save(final_path, data)

        #Check to See if File Exists
        if os.path.isfile(final_path) == True:
            return "File Saved to Numpy File"
        elif os.path.isfile(final_path) == False:
            return "File Not Saved, Try Again"



def load_df_np(filename, type):
    """
    Locates the stored file as either a pandas dataframe or numpy array and loads
    the item into the workspace.
    """

    if type == "pandas":
        #Set Pathway
        parent = find_parent_filepath()
        path_to_pkl = parent + "/train_storage/"
        final_path = path_to_pkl + filename + ".pkl"

        #Check to See if File Exists and Import if Found
        if os.path.isfile(final_path) == True:
            print("Pickle Found, Importing to Pandas DataFrame")
            df = pd.read_pickle(final_path)
            return df

        elif os.path.isfile(final_path) == False:
            return "Check File Path and Try Again"

    elif type == "numpy":
        #Set Pathway
        parent = find_parent_filepath()
        path_to_pkl = parent + "/train_storage/"
        final_path = path_to_pkl + filename + ".npy"

        #Check to See if File Exists and Import if Found
        if os.path.isfile(final_path) == True:
            print("Numpy File Found, Importing to Numpy Array")
            np_array = np.load(final_path)
            return np_array

        elif os.path.isfile(final_path) == False:
            return "Check File Path and Try Again"
