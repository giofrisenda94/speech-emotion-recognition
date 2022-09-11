import gcloud
from gcloud import storage
import numpy as np
import pandas as pd
import os


#Need to have Credentials.json on you computer
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= "credentials.json"

#Push Dataframe to Bucket
def push_to_bucket(data):
    """
    Takes numpy and pushes it to the google cloud bucket
    """


    fil_name = "processed_features/features"

    np.save(fil_name, data)

    path = "processed_features/features.npy"

    # Initialise a client
    client = storage.Client("lewagon-bootcamp-355711")

    # Create a bucket object for our bucket
    bucket = client.get_bucket("220905-spe-rec")

    # Create a blob object from the filepath
    blob = bucket.blob("Prediction-Features")

    # Upload the file to a destination
    blob.upload_from_filename(path)
