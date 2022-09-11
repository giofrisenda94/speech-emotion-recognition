import sys
import pathlib
sys.path.append("/".join(str(pathlib.Path().absolute()).split("/")[0:-1] + ["functions"]))

import os
import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
import streamlit.components.v1 as components
import requests
from functions.recording_functions import store_wav_file
from functions.recording_functions import recording_feature_extraction
from functions.gloud_functions import push_to_bucket
import requests


# DESIGN implement changes to the standard streamlit UI/UX
st.set_page_config(layout="centered", page_title="Speech Emotion Recognition App")

st.title("Speech Emotion Recognition App")

st.markdown("""
            This App uses a Recurrent Neural Network(RNN) method (Long Short-Term Memory (LSTM) to predict expressed emotions from a live audio.
            """)


st.write("")
st.write("")


def audiorec():
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    # Custom REACT-based component for recording client audio in browser
    build_dir = os.path.join(parent_dir, "st_audiorec/frontend/build")
    # specify directory and initialize st_audiorec object functionality
    st_audiorec = components.declare_component("st_audiorec", path=build_dir)

    # STREAMLIT AUDIO RECORDER Instance
    val = st_audiorec()
    # web component returns arraybuffer from WAV-blob

    if isinstance(val, dict):  # retrieve audio data
        with st.spinner('retrieving audio-recording...'):
            ind, val = zip(*val['arr'].items())
            ind = np.array(ind, dtype=int)  # convert to np array
            val = np.array(val)             # convert to np array
            sorted_ints = val[ind]
            stream = BytesIO(b"".join([int(v).to_bytes(1, "big") for v in sorted_ints]))
            wav_bytes = stream.read()

        # wav_bytes contains audio data in format to be further processed
        # display audio data as received on the Python side
        #st.audio(wav_bytes, format='audio/wav')

        return wav_bytes



st.subheader("Record your Audio")
#Create WAV Bytes from recorded Audio File
wav_bytes = audiorec()


if wav_bytes != None:

    st.write("")
    st.write("")

    st.subheader("Audio File Recorded, Run the Prediction")

    selection_list = []

    #Generate WAV File and Store Either Locally or to Temp File
    file_path = store_wav_file("temp", wav_bytes)

    if st.button("Predict") == True:
        selection_list.append(1)


    if selection_list != []:
        if selection_list[0] == 1:

            #Grab Audio File
            audio_file = file_path

            #Process Audio File
            X = recording_feature_extraction(audio_file)


            # #Run Local Model
            # model = load_model("model8723")


            # prediction = model.predict(X)


            # #Set Emotions List
            # emotions = {
            #     0 : 'Neutral',
            #     1 : 'Calm',
            #     2 : 'Happy',
            #     3 : 'Sad',
            #     4 : 'Angry',
            #     5 : 'Fearful',
            #     6 : 'Disgust',
            #     7 : 'Suprised'
            # }
            # emo_list = pd.Series(emotions.values())


            # #Process Predictions and Produce Results
            # predict_df = pd.DataFrame(prediction).T
            # predict_df["emotion"] = emo_list
            # predict_df = predict_df.rename(columns={0: 'percentage'})
            # predict_df.rename(columns = {"emotion": "EMOTIONS", "percentage":"PERCENTAGE"}, inplace = True)

            # predict_df = predict_df[predict_df['EMOTIONS'] != "Fearful"]

            # st.write("")

            # #Write Highest Value
            # max_predict = predict_df.sort_values("PERCENTAGE", ascending = False)
            # max_predict.reset_index(inplace = True)
            # max_predict = max_predict.head(3)

            # display_emotions = {
            #     "Neutral":"Neutralness",
            #     "Calm":"Calmness",
            #     "Happy":"Happiness",
            #     "Sad":"Sadness",
            #     "Angry": "Anger",
            #     "Fearful":"Fear",
            #     "Disgust":"Disgust",
            #     "Suprised":"Suprise"
            # }

            # max_predict["EMOTIONS"] = max_predict["EMOTIONS"].map(display_emotions)



            # message = f"Your voice shows signs of **{max_predict['EMOTIONS'].loc[0]}** with a hint of **{max_predict['EMOTIONS'].loc[1]}** and a sparkling of **{max_predict['EMOTIONS'].loc[2]}**"


            #Display the Graph
            #st.bar_chart(predict_df, x ='EMOTIONS', y ='PERCENTAGE')


            #Load to GCloud Bucket
            push_to_bucket(X)


            #Send Prediction Request
            url = "https://speech-emotion-recognition-uulxozpkpq-ez.a.run.app/predict"
            response = requests.get(url).json()


            url = "https://speech-emotion-recognition-uulxozpkpq-ez.a.run.app/predict"
            response = requests.get(url).json()

            emotions = [
            'Neutral',
            'Calm',
            'Happy',
            'Sad',
            'Angry',
            'Fearful',
            'Disgust',
            'Suprised']

            pred = response["greeting"]

            pred = pred.replace("\n ", '').replace("[[", '').replace("]]", '')

            pred_np = np.fromstring(pred, dtype = "float", sep=' ')

            pred_pd = pd.DataFrame(pred_np)

            pred_pd["EMOTIONS"] = emotions

            predict_df = pred_pd

            predict_df = predict_df.rename(columns={0: 'percentage'})
            predict_df.rename(columns = {"emotion": "EMOTIONS", "percentage":"PERCENTAGE"}, inplace = True)

            predict_df = predict_df[predict_df['EMOTIONS'] != "Fearful"]

            max_predict = predict_df.sort_values("PERCENTAGE", ascending = False)
            max_predict.reset_index(inplace = True)
            max_predict = max_predict.head(3)

            display_emotions = {
                "Neutral":"Neutralness",
                "Calm":"Calmness",
                "Happy":"Happiness",
                "Sad":"Sadness",
                "Angry": "Anger",
                "Fearful":"Fear",
                "Disgust":"Disgust",
                "Suprised":"Suprise"
            }

            max_predict["EMOTIONS"] = max_predict["EMOTIONS"].map(display_emotions)

            st.write("")
            st.write("")

            message = f"Your voice shows signs of **{max_predict['EMOTIONS'].loc[0]}** with a hint of **{max_predict['EMOTIONS'].loc[1]}** and a sparkling of **{max_predict['EMOTIONS'].loc[2]}**"

            st.write(message)

            #Display the Graph
            st.bar_chart(predict_df, x ='EMOTIONS', y ='PERCENTAGE')
