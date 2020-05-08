#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.io.wavfile
from scipy.fftpack import dct
import numpy as np
import os
from tqdm import tqdm
import pickle
from random import randint
from math import ceil, floor

TRAINING_PATH = "../../training/"
NOISE_PATH = "../../_background_noise_/"
VANILLA_DATA_NAME = '../pickle_files/mfcc_noise_data_'
BLOCK_SIZE = 512
CEPSTRAL = 12

# In[2]:
def load_all_noises():
    all_noises = os.listdir(NOISE_PATH)
    audios = []
    for noise in all_noises:
        _, audio = scipy.io.wavfile.read(NOISE_PATH + noise)
        audio = audio[:16000]
        audio = audio * 0.002
        audios.append(audio)
    return audios

def add_noise(input_audio):
    index = randint(0,4)
    result = input_audio + NOISES[index]
    return result

'''
Main Feature Function
'''
def preprocess_audio(audio):
    beg = audio[0]
    non_scaled = audio[1:]
    scaled = 0.97 * audio[:-1]
    result = [beg]
    result.extend(non_scaled - scaled)
    return np.array(result)

    
def split_into_frames(pre_audio, rate):
    fl = int(round(0.025 * rate))
    fs = int(round(0.01 * rate))
    
    diff = len(pre_audio) - fl
    norm = abs(diff)/fs
    fc = int(ceil(norm))
    
    audio = pre_audio.tolist()
    audio.extend([0 for i in range(fc * fs + fl - len(pre_audio)) ])
    audio = np.array(audio)
    
    
    beg_part = np.arange(0, fl)
    end_part = (fc, 1)
    beg_index = np.tile(beg_part, end_part)
    
    upper_lim = fc * fs
    beg_part = np.arange(0, upper_lim, fs)
    end_part = (fl,1)
    end_index = np.tile(beg_part, end_part).T
    index_of_frames = beg_index + end_index
    
    frames = audio[index_of_frames]
    
    frames *= np.hamming(fl)
    
    return frames

def fourier_transform(frames):
    frames = abs(np.fft.rfft(frames, BLOCK_SIZE))
    frames_square = frames ** 2
    norm = 1.0 / BLOCK_SIZE
    frames_final  = norm * frames_square
    return frames_final

def get_binning_arrays(rate):
    high = (2595 * np.log10(1 + (rate / 2) / 700))  # Convert Hz to Mel
    points = (700 * (10**(np.linspace(0, high , 42) / 2595) - 1))  # Convert Mel to Hz
    norm = (BLOCK_SIZE + 1) / rate
    return np.floor(norm * points)


def generate_filter_banks(frames, rate):
    binning_array = get_binning_arrays(rate)
    
    temp_array = np.zeros((40, int(np.floor(BLOCK_SIZE / 2 + 1))))
    
    for bin_index in range(1, 41):
        low_index = bin_index - 1
        high_index = bin_index + 1

        for pos in range(int(binning_array[low_index]), int(binning_array[bin_index])):
            numer = pos - binning_array[low_index]
            denom = binning_array[bin_index] - binning_array[low_index]
            temp_array[low_index, pos] = numer / denom
        
        for pos in range(int(binning_array[bin_index]), int(binning_array[high_index])):
            numer = binning_array[high_index] - pos
            denom = binning_array[high_index] - binning_array[bin_index]
            temp_array[low_index, pos] = numer / denom
          
    filter_banks = np.dot(frames, temp_array.T)
    
    filter_banks[filter_banks == 0] = np.finfo(float).eps  
    
    return 20 * np.log10(filter_banks)

def get_mfcc_norm(cols):
    array = np.pi * np.arange(cols) / 22
    sin_array = np.sin(array)
    return 11 * sin_array + 1

def get_mfcc_feature(audio, rate, mean_norm = True):
    
    pre_audio = preprocess_audio(audio)
    
    frames = split_into_frames(pre_audio, rate)
    
    frames = fourier_transform(frames)
    
    filter_banks = generate_filter_banks(frames, rate)
    
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (CEPSTRAL + 1)]
    
    norm = get_mfcc_norm(mfcc.shape[1])
    
    mfcc *= norm
    
    if mean_norm:
        diff = np.mean(mfcc, axis=0) + 1e-8
        mfcc -= diff
    
    return mfcc


# In[4]:


def save_vanilla_mfcc_features():
    X = []
    y = []
    all_digits = ['zero', 'one', 'two']
    # all_digits = ['three','four', 'five']
    # all_digits = ['six','seven','eight','nine']
    for digit in all_digits:
        print (digit.upper())
        path_till_now = TRAINING_PATH + digit + "/"
        all_audios = os.listdir(path_till_now)
        for audio_file in tqdm(all_audios):
            rate, audio = scipy.io.wavfile.read(path_till_now + audio_file)
            length_of_audio = len(audio)
            if length_of_audio < 16000:
                audio = np.pad(audio, (0, 16000 - length_of_audio), mode = 'constant')
            elif length_of_audio > 16000:
                audio = audio[:16000]
            feature = get_mfcc_feature(audio, rate)
            X.append(feature)
            y.append(digit)
            noisy_audio = add_noise(audio)
            noisy_feature = get_mfcc_feature(noisy_audio, rate)
            X.append(noisy_feature)
            y.append(digit)
    data = [X,y]
    with open(VANILLA_DATA_NAME + "1.pkl", 'wb') as f:
        pickle.dump(data, f)


# In[ ]:


NOISES = load_all_noises()
save_vanilla_mfcc_features()
