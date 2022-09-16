from fastapi import FastAPI
import numpy as np
from gcloud import storage
from tensorflow.keras import models


app = FastAPI()


#Initialize API
@app.get("/")
def hello():
    return "Hello"



@app.get("/predict")

def pred():

    # Initialise a client
    client = storage.Client("enter_client_name")

    #Create a bucket object for our bucket
    bucket = client.get_bucket("enter_bucket_name")

    # Create a blob object from the filepath
    features_blob = bucket.blob("Prediction-Features")

    # Load Numpy
    features_blob.download_to_filename("test.npy")

    # Load Features
    features = np.load("test.npy")

    # Load Model
    model = models.load_model("model_final.h5")

    # Run Predictions
    pred = model.predict(features)

    strings = str(pred)

    #Decode Prediction to Human Readable Format

    return dict(greeting=strings)
