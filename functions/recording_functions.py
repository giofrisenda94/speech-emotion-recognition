import librosa
import numpy as np
import noisereduce as nr
from pydub import AudioSegment, effects
from datetime import datetime
import tempfile
from functions.path_functions import find_parent_filepath
from functions.local_model_functions import pitch_process, shift_process, stretch_process, add_noise
from functions.local_model_functions import extract_process
from functions.local_model_functions import export_process





#Grab Raw Audio File from Interface and Store Locally or in Temp File
def store_wav_file(location, wav_bytes):
    """
    Stores the recording made in the interface and returnt the filepath either to a
    local drive or to a temp location, specify where by passing in the value location
    as either local or temp

    location values
        local: File is stored in a folder named "recordings"
        temp: File is stored in a temp location on the computer
    """

    file_location = ""

    location = location

    if location == "local":
        #Create a timestamp for the file
        time_label = datetime.now().strftime("%Y%m%d%H%M%S")

        #Create Lable for File
        file_label = "recording-" + time_label + ".wav"

        #Store File in Recordings Folder
        with open(f'streamlit_app/recordings/{file_label}', mode='bx') as f:
            f.write(wav_bytes)

        #Find Path to Speech-Emotion-Recognition Folder
        parent_filepath = find_parent_filepath()

        #Create Final Filepath
        local_filepath = parent_filepath + "/streamlit_app/recordings/" + file_label

        #Set Final Filepath
        file_location = local_filepath


    elif location == "temp":
        #Create Temporary Directory and Temporary Filepath
        temp_dir = tempfile.tempdir
        temp_filepath = tempfile.mkdtemp(dir = temp_dir)

        #Store in Temporary Location
        with open(f'{temp_filepath}.wav', mode='bx') as f:
            f.write(wav_bytes)

        file_location = f'{temp_filepath}.wav'


    return file_location




#Grab Audio File from Streamlit and Process into Ravdess Numpy Array
def ravdess_recording_processing(recording_filepath):
    """
    Takes the processed .wav file from the store_wav_file step and converts it
    to a processed dataframe needed for feature extraction
    """

    #Create Librosa Item from Audio Recording
    x, sr = librosa.load(recording_filepath, sr = None)

    #Normalize Librosa Item
    normalizedsound = effects.normalize(AudioSegment.from_file(recording_filepath), headroom = 5.0)

    #Create Numpy Array from Normalized file
    normal_x = np.array(normalizedsound.get_array_of_samples(), dtype = 'float32')

    #Trim Numpy Array
    xt, index = librosa.effects.trim(normal_x, top_db = 30)

    #Reduce Noise and Output Final Numpy Arrray
    final_x = nr.reduce_noise(xt, sr=sr)

    return final_x, sr


#Extract Features from Processed Ravdess Audio File
def recording_feature_extraction(audio_file):
    """
    Takes a recording and creates and processes a final Numpy Array

    rms, zrc, mfcc, tonnetz = recording_feature_extraction()
    """

    total_length = 250000
    frame_length = 2048
    hop_length = 512

    _, sr = librosa.load(path = audio_file, sr = None)

    rawsound = AudioSegment.from_file(audio_file)

    normalizedsound = effects.normalize(rawsound, headroom = 0)

    normal_x = np.array(normalizedsound.get_array_of_samples(), dtype = 'float32')

    xt, index = librosa.effects.trim(normal_x, top_db=30)

    padded_x = np.pad(xt, (0, total_length-len(xt)), 'constant')

    final_x = nr.reduce_noise(padded_x, sr=sr)

    f1 = librosa.feature.rms(final_x, frame_length=frame_length, hop_length=hop_length, center=True, pad_mode='reflect').T # Energy - Root Mean Square
    f2 = librosa.feature.zero_crossing_rate(final_x, frame_length=frame_length, hop_length=hop_length,center=True).T # ZCR
    f3 = librosa.feature.mfcc(final_x, sr=sr, S=None, n_mfcc=13, hop_length = hop_length).T # MFCC

    X = np.concatenate((f1, f2, f3), axis = 1)

    X = np.expand_dims(X, axis=0)

    return X
