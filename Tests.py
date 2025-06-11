
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy import fft
import audio_functions as af

x1 = np.array([0, 0, 0, 1, 1, 0, 1])
x2 = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1])


N = len(x1) + len(x2) - 1  # N = 5

# Zero padding
x1_padded = np.pad(x1, (0, N - len(x1)))
x2_padded = np.pad(x2, (0, N - len(x2)))

conv = np.convolve(x1, x2)


x1_fft = fft.fft(x1_padded)
x2_fft = fft.fft(x2_padded)

corr_1 = signal.correlate(x1, x2, mode="full")
corr_2 = x2_fft * np.conjugate(x1_fft)
corr_2 = np.round(np.real(fft.ifft(corr_2)))
corr_2 = np.fft.fftshift(corr_2)
corr_2 = np.roll(corr_2, -1)

corr_3 = af.cross_corr(x1, x2, fs=10, mode = "Classic")

print(f"Señal 1: {x1}")
print("")
print(f"Señal 2: {x2}")
print(f"Señal 1 Padded: {x1_padded}")
print("")
print(f"Señal 2 Padded: {x2_padded}")
print("")
#print(f"Convolución: {conv}")
#print("")
#print(f"FFT 1: {x1_fft}")
#print("")
#print(f"FFT 2: {x2_fft}")
#print("")
print(f"Correlación en muestras: {corr_1}")
print("")
print(f"Longitud de correlación en muestras: {len(corr_1)}")
print("")
print(F"Correlación en frecuencia: {corr_2}")
print("")
print(f"Longitud de correlación en frecuencia: {len(corr_2)}")
print("")
print(f"Correlacion con funcion cross_corr: {corr_3}")
print("")
print(f"Longitud: {len(corr_3)}")














