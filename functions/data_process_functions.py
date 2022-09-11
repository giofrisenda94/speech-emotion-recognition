import os
import glob
import pandas as pd
import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr
from pydub import AudioSegment, effects
from datetime import datetime
import tempfile
from functions import path_functions as pf
from functions import feature_extraction as fe
from functions.path_functions import find_parent_filepath
#from functions import models as mo


FRAME_LENGTH=2048
HOP_LENGTH=512
FEATURES = ["rms", "zero_crossing_rate", "mfcc", "tonnetz"]

def ravdess_df(ravdess_path):
    """Create a data frame from Ravdess dataset"""

    file_id = []
    emotions = []
    actor_gender = []


    for file in glob.glob(ravdess_path):

        file_name = os.path.basename(file)

        file_id.append(str(file_name))

        emotions.append(int(file_name.split("-")[2]))

        actor = int(file_name.split("-")[6].split(".")[0])
        if actor % 2 == 0:
            actor_gender.append(0)
        else:
            actor_gender.append(1)

    ravdess_dict = {'file_id':file_id, 'emotions':emotions, 'actor_gender': actor_gender}


    return pd.DataFrame(ravdess_dict)



def tess_df(tess_path):
    """Create a data frame from Tess dataset"""

    file_id = []
    emotions = []
    actor_gender = []

    for file in glob.glob(tess_path):

        file_name = os.path.basename(file)

        file_id.append(str(file_name))

        emotions.append(file_name.split("_")[2].split('.')[0])

        actor_gender.append(0)

    emotions_code = []
    for i in emotions:
        if i == 'angry':
            emotions_code.append(5)
        elif i == 'disgust':
            emotions_code.append(7)
        elif i == 'fear':
            emotions_code.append(6)
        elif i == 'neutral':
            emotions_code.append(1)
        elif i == 'ps':
            emotions_code.append(8)
        elif i == 'sad':
            emotions_code.append(4)
        elif i == 'happy':
            emotions_code.append(3)

    tess_dict = {'file_id':file_id, 'emotions':emotions_code, 'actor_gender': actor_gender}

    return pd.DataFrame(tess_dict)



def crema_df(crema_path):
    """Create a data frame from Crema dataset"""

    file_id = []
    emotions = []
    actor_gender = []

    female = [1002,1003,1004,1006,1007,1008,1009,1010,1012,1013,1018,1020,1021,1024,1025,1028,1029,1030,1037,1043,1046,1047,1049,
          1052,1053,1054,1055,1056,1058,1060,1061,1063,1072,1073,1074,1075,1076,1078,1079,1082,1084,1089,1091]

    for file in glob.glob(crema_path):

        file_name = os.path.basename(file)

        file_id.append(str(file_name))

        emotions.append(file_name.split("_")[2])

        if int(file_name.split("_")[0]) in female:
            actor_gender.append(0)
        else:
            actor_gender.append(1)


    emotions_code = []
    for i in emotions:
        if i == 'ANG':
            emotions_code.append(5)
        elif i == 'DIS':
            emotions_code.append(7)
        elif i == 'FEA':
            emotions_code.append(6)
        elif i == 'NEU':
            emotions_code.append(1)
        elif i == 'HAP':
            emotions_code.append(3)
        elif i == 'SAD':
            emotions_code.append(4)

    crema_dict = {'file_id':file_id, 'emotions':emotions_code, 'actor_gender': actor_gender}

    return pd.DataFrame(crema_dict)



def savee_df(savee_path):
    """Create a data frame from Savee dataset"""

    file_id = []
    emotions = []
    actor_gender = []

    for file in glob.glob(savee_path):

        file_name = os.path.basename(file)

        file_id.append(str(file_name))

        actor_gender.append(1)

        if file_name.split('_')[-1].split('.')[0][0] == 's':
            emotions.append(file_name.split('_')[-1].split('.')[0][:2])
        else:
            emotions.append(file_name.split('_')[-1].split('.')[0][0])


    emotions_code = []
    for i in emotions:
        if i == 'a':
            emotions_code.append(5)
        elif i == 'd':
            emotions_code.append(7)
        elif i == 'f':
            emotions_code.append(6)
        elif i == 'n':
            emotions_code.append(1)
        elif i == 'h':
            emotions_code.append(3)
        elif i == 'sa':
            emotions_code.append(4)
        elif i == 'su':
            emotions_code.append(8)

    savee_dict = {'file_id':file_id, 'emotions':emotions_code, 'actor_gender': actor_gender}

    return pd.DataFrame(savee_dict)



#Tess Process Add_Noise
def tess_add_noise(data):
    #noise
    noise_value = 0.015 * np.random.uniform() * np.amax(data)
    data = data + noise_value * np.random.normal(size=data.shape[0])

    return data


#Tess Stretch
def tess_stretch_process(data,rate=0.8):
    #stretch
    return librosa.effects.time_stretch(data,rate)

#Tess Shift
def tess_shift_process(data):
    #shift
    shift_range = int(np.random.uniform(low=-5,high=5) * 1000)
    return np.roll(data,shift_range)

#Tess Pitch
def tess_pitch_process(data,sampling_rate,pitch_factor=0.7):
    #pitch
    return librosa.effects.pitch_shift(data,sampling_rate,pitch_factor)



#Tess Extract Features
def tess_extract_process(data, sample_rate = 1):

    output_result = np.array([])
    mean_zero = np.mean(librosa.feature.zero_crossing_rate(y=data).T,axis=0)
    output_result = np.hstack((output_result,mean_zero))

    stft_out = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft_out,sr=sample_rate).T,axis=0)
    output_result = np.hstack((output_result,chroma_stft))

    mfcc_out = np.mean(librosa.feature.mfcc(y=data,sr=sample_rate).T,axis=0)
    output_result = np.hstack((output_result,mfcc_out))

    root_mean_out = np.mean(librosa.feature.rms(y=data).T,axis=0)
    output_result = np.hstack((output_result,root_mean_out))

    mel_spectogram = np.mean(librosa.feature.melspectrogram(y=data,sr=sample_rate).T,axis=0)
    output_result = np.hstack((output_result,mel_spectogram))

    return output_result


if __name__=="__main__":
    ##1 import the path
    root_path = pf.find_parent_filepath()
    ##2 run the dataframe creation from dataset
    ravdess_dataframe = ravdess_df(root_path)
    ##rd_df, path = fe.ravdess_generation()
    #3 feature extraction of data
    file_id_list, processed_sound_list, sr = fe.ravdess_preprocessing(ravdess_dataframe, root_path)
    extracted_feature_df=  fe.ravdess_feature_extraction(file_id_list, processed_sound_list, sr, FRAME_LENGTH, HOP_LENGTH)
    #4 model run
    X_train, X_test, y_train_class, y_test_class = mo.ravdess_x_y_train_test(FEATURES)
    model = mo.lstm_4_model(X = X_train)
    model = mo.fit_model(model, X_train, y_train_class)
    model.score(X_test, y_test_class, scoring=["categorical_accuracy"])
