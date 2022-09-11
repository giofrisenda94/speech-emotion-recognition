import os
import numpy as np
import pandas as pd

import pydub

from pydub import AudioSegment, effects
import librosa
import soundfile as sf
import noisereduce as nr

from functions.data_process_functions import ravdess_df
from functions.path_functions import find_parent_filepath

import glob



def ravdess_generation():
    path = f'{find_parent_filepath()}/raw_data/Ravdess/*/*.wav'
    rd_df = pd.DataFrame(ravdess_df(path))
    return rd_df, path


def ravdess_preprocessing(rd_df, path):

    file_id_list= []
    processed_sound_list = []
    max_len = 228864 #calculated beforehand

    for file in glob.glob(path):
        x, sr = librosa.load(file, sr = None)
        normalizedsound = effects.normalize(AudioSegment.from_file(file), headroom = 5.0)
        normal_x = np.array(normalizedsound.get_array_of_samples(), dtype = 'float32')
        xt, index = librosa.effects.trim(normal_x, top_db = 30)
        padded_x = np.pad(xt, (0, max_len-len(xt)), 'constant')
        final_x = nr.reduce_noise(padded_x, sr=sr)
        file_id_list.append(f"{file[-24:]}")
        processed_sound_list.append(final_x)


    #feature_df = pd.DataFrame(list(zip(file_id_list, processed_sound_list, sample_rate_list)), columns =["file_id", "processed_sound", "sample_rate"])
    #preprocessed_df = pd.merge(ravdess_df, feature_df, on=["file_id"])
    return file_id_list, processed_sound_list, sr



def ravdess_feature_extraction(file_id_list, processed_sound_list, sr, frame_length, hop_length):

    rms_feature_list = []
    zero_crossing_rate_list = []
    mfcc_list = []
    tonnetz_list = []

    for i in processed_sound_list:
        rms_feature = librosa.feature.rms(i, frame_length=frame_length, hop_length=hop_length)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(i, frame_length=frame_length, hop_length=hop_length) # Zero Crossed Rate (ZCR)
        mfcc = librosa.feature.mfcc(i, sr=sr, S=None, n_mfcc=25, hop_length = hop_length) # MFCCs
        tonnetz = librosa.feature.tonnetz(i)

        rms_feature_list.append(rms_feature)
        zero_crossing_rate_list.append(zero_crossing_rate)
        mfcc_list.append(mfcc)
        tonnetz_list.append(tonnetz)

    extracted_feature_df = pd.DataFrame(list(zip(file_id_list, rms_feature_list, zero_crossing_rate_list, mfcc_list, tonnetz_list)), columns =["file_id", "rms", "zero_crossing_rate", "mfcc", "tonnetz"])

    return extracted_feature_df
