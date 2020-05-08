#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm
from random import randint

TRAINING_PATH = "../../training/"
VANILLA_DATA_NAME = '../pickle_files/spectogram_noise_data_'
NOISE_PATH = "../../_background_noise_/"
BLOCK_SIZE = 512
OVERLAP = 256


# In[2]:

def load_all_noises():
    all_noises = os.listdir(NOISE_PATH)
    audios = []
    for noise in all_noises:
        audio, _ = librosa.load(NOISE_PATH + noise)
        audio = audio[:16000]
        audio = audio * 0.002
        audios.append(audio)
    return audios

def add_noise(input_audio):
    index = randint(0,4)
    result = input_audio + NOISES[index]
    return result

'''
Helper Functions
'''
def get_fourier_coefficient(array,i,length,sub_positions):
    exp_constant = 1j * 2
    exp_num = exp_constant * np.pi * sub_positions * i
    exp_denom = length
    exp_eq = exp_num / exp_denom
    exp_value = np.exp(exp_eq)
    array = array * exp_value
    s = np.sum(array)
    norm = abs(s/length) * 2
    return(norm)

def gen_position_array(upper_limit):
    diff = BLOCK_SIZE - OVERLAP
    value = 0
    array = []
    for i in range(0, upper_limit):
        if (value + BLOCK_SIZE) < upper_limit:
            array.append(value)
            value += diff
    return array


# In[3]:


'''
Main Feature Extractor
'''
def get_spectogram_feature(audio):
    length_complete = len(audio)
    
    positions = gen_position_array(length_complete)
    
    result = []
    
    for pos in positions:
        sub_array = audio[pos:pos + BLOCK_SIZE]
        length = len(sub_array)
        half_length = int(length/2)
        
        window = []
        for i in range(half_length):
            sub_positions = np.arange(0,length,1)
            window.append(get_fourier_coefficient(sub_array, i, length, sub_positions))
            
        result.append(window)
        
    transpose = np.array(result).T
    
    return 10 * np.log10(transpose)


# In[4]:


def save_vanilla_spectogram_features():
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
            audio, _ = librosa.load(path_till_now + audio_file)
            length_of_audio = len(audio)
            if length_of_audio < 16000:
                audio = np.pad(audio, (0, 16000 - length_of_audio), mode = 'constant')
            elif length_of_audio > 16000:
                audio = audio[:16000]
            feature = get_spectogram_feature(audio)
            X.append(feature)
            y.append(digit)
            noisy_audio = add_noise(audio)
            noisy_feature = get_spectogram_feature(noisy_audio)
            X.append(noisy_feature)
            y.append(digit)
    data = [X,y]
    with open(VANILLA_DATA_NAME + "1.pkl", 'wb') as f:
        pickle.dump(data, f)


# In[ ]:

NOISES = load_all_noises()
save_vanilla_spectogram_features()

