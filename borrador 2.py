import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy import fft
import audio_functions as auf
import plot


iters = 5
names = {}
for i in range(iters):
    names[f"name{i}"] = f"mic i"

print(names)