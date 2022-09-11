import numpy as np
from functions.feature_extraction import ravdess_generation
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from functions.feature_extraction import ravdess_preprocessing
from functions.feature_extraction import ravdess_feature_extraction


def ravdess_x_y_train_test(features = ["rms", "zero_crossing_rate", "mfcc", "tonnetz"], train_size = .7):

    #Find File Path
    rd_df, path = ravdess_generation()

    #Create Target Emotions
    emotions = np.array(rd_df["emotions"])

    #Create File IDs, Processed Sounds, and SR Value
    file_id_list, processed_sound_list, sr = ravdess_preprocessing(rd_df, path)

    #Create Final Dataframe with Information
    final_df = ravdess_feature_extraction(file_id_list, processed_sound_list, sr, 2048, 512)

    selection = ["file_id"]
    for item in features:
        selection.append(item)

    #Select from DataFrame Features Selected
    X = final_df[selection]

    #Create Target
    y = emotions
    y_class = to_categorical(LabelEncoder().fit_transform(y))


    #Create Train Test Split
    X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, train_size = train_size)


    return X_train, X_test, y_train_class, y_test_class
