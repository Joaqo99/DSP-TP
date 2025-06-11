import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy import fft
import audio_functions as auf


x1 = np.array([1, 0, 1])
x2 = np.array([1, 0, 1, 0, 0])


N = len(x1) + len(x2) - 1

conv = np.convolve(x1, x2)
corr = signal.correlate(x1, x2, mode="full")
lags = signal.correlation_lags(len(x1), len(x2), mode="full")
lag = lags[np.argmax(corr)]

print("No padding")
print(f"Convolution (numpy): {conv}")

print("Padding")
## Zero padding
x1_padded = np.pad(x1, (0, N - len(x1)))
x2_padded = np.pad(x2, (0, N - len(x2)))
print(f"X1 Padded: {x1_padded}")
print(f"X2 Padded: {x2_padded}")
x1_fft = fft.fft(x1_padded)
x2_fft = fft.fft(x2_padded)

conv_2 = fft.ifft(x1_fft*x2_fft)
print(f"Convolution (fft): {np.real(conv_2).astype(int)}")

cross_pow_spectrum = np.conjugate(x1_fft)*x2_fft
corr_2 = fft.ifft(cross_pow_spectrum)
corr_2 = np.real(corr_2).astype(int)
corr_3 = np.roll(corr_2, -1)

print(f"Correlation 1: {corr}")
print(f"Correlation 2: {corr_2}")
print(f"Correlation 3: {corr_3}")
print(f"Lag (scipy): {lag}")
