import numpy as np
import librosa
import pandas as pd

def pitch_process(data,sampling_rate,pitch_factor=0.7):
    #pitch
    return librosa.effects.pitch_shift(data,sampling_rate,pitch_factor)


def shift_process(data):
    #shift
    shift_range = int(np.random.uniform(low=-5,high=5) * 1000)
    return np.roll(data,shift_range)


def stretch_process(data,rate=0.8):
    #stretch
    return librosa.effects.time_stretch(data,rate)


def add_noise(data):
    #noise
    noise_value = 0.015 * np.random.uniform() * np.amax(data)
    data = data + noise_value * np.random.normal(size=data.shape[0])

    return data



def extract_process(data, sample_rate = 1):

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



def export_process(path):

    data,sample_rate = librosa.load(path,duration = 2.5,offset = 0.6)

    output_1 = extract_process(data, sample_rate)
    result = np.array(output_1)

    noise_out = add_noise(data)
    output_2 = extract_process(noise_out)
    result = np.vstack((result,output_2))

    new_out = stretch_process(data)
    strectch_pitch = pitch_process(new_out,sample_rate)
    output_3 = extract_process(strectch_pitch)
    result = np.vstack((result,output_3))

    return result
