import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy import fft
import audio_functions as auf
import plot

def get_audio_time_array_1(audio, fs):
    """
    Returns audio time array
    Input:
        - audio: array type object.
        - fs: Int type object. Sample rate.
    Output:
        - duration: int type object. Audio duration
        - time_array: array type object.
    """
    #error handling
    if  type(audio) != np.ndarray:
        raise ValueError("audio must be a ndarray")
    if type(fs) != int:
        raise ValueError("fs must be int")
    
    #features
    duration = len(audio) / fs
    time_array = np.linspace(0, duration, len(audio))

    return duration, time_array

def get_audio_time_array_2(audio, fs):
    """
    Returns audio time array
    Input:
        - audio: array type object.
        - fs: Int type object. Sample rate.
    Output:
        - duration: int type object. Audio duration
        - time_array: array type object.
        
    Raices:
        - TypeError: if audio is not an array 
        - TypeError: if fs is not an int
    """

    duration = audio.size // fs
    time_array = np.linspace(0, duration, audio.size)

    return duration, time_array

fs = 48000

audio, _ = auf.load_audio("audios_anecoicos/p336_001.wav")

_, n1 = get_audio_time_array_1(audio, fs)
_, n2 = get_audio_time_array_2(audio, fs)

plot.unit_plot({"array":n1, "label":"n1"}, {"array":n2, "label":"n2"})