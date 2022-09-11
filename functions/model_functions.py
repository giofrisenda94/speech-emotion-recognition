import numpy as np
import pandas as pd
import tensorflow
from functions.path_functions import find_parent_filepath

EarlyStopping = tensorflow.keras.callbacks.EarlyStopping()



def save_model(model, filename):
    """
    Save the model to the a folder with the trained models.  Takes the model in arg and name of the saved model.
    """
    model_json = model.to_json()

    parent = find_parent_filepath()
    storage = parent + "/trained_models/"

    saved_model_path =  storage + filename +'.json'
    saved_weights_path = storage + filename + '_weights.h5'


    with open(saved_model_path, "w") as json_file:
        json_file.write(model_json)

    model.save_weights(saved_weights_path)
    print("Saved to Trained Models Folder")



def load_model(filename, loss = "categorical_crossentropy", optimizer = "adam", metrics = ['accurracy']):
    """
    Loads the models stored in the trained models folder and returns the trained tensorflow model
    and compiles it at the same time.
    """

    parent = find_parent_filepath()
    storage = parent + "/trained_models/"

    saved_model_path =  storage + filename +'.json'
    saved_weights_path = storage + filename + '_weights.h5'

    with open(saved_model_path , 'r') as json_file:
        json_savedModel = json_file.read()

    # Loading the model architecture, weights
    model = tensorflow.keras.models.model_from_json(json_savedModel)
    model.load_weights(saved_weights_path)

    # Compiling the model with similar parameters as the original model.
    model.compile(loss=loss,
                    optimizer=optimizer,
                    metrics=metrics)

    return model




#Fit a Model
def fit_model(model, X_train, y_train, epochs = 300, batch_size = 256, validation_split = 0.3, patience = 20, verbose = 1):
    """
    Takes a previously created model and fits it to the training data.  Pass a previously
    created model with the training data to fit the model
    """

    #Create Early Stopping Metric
    es = EarlyStopping(patience=patience, restore_best_weights=True)

    model.fit(X_train,
              y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            callbacks=[es]
            )

    return model



def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model against the X_test and y_test.
    Returns the evaluation output.  Requires the model, X_test and y_test frame.
    """

    eval = model.evaluate(X_test, y_test, verbose = 1)

    return eval




def model_predict(model, input):
    """
    Takes the input audio recording from streamlit and plugs it into the model
    to predict emotions.
    """

    predict = model.predict()
    return predict
