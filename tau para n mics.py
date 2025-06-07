import numpy as np
import plot
import audio_functions as af
from scipy.signal import correlate
from scipy import signal

#-----------------------------------------------------------------------
c = 343    
fs=44100
duration = 0.1      # duracion de la señal

t = np.linspace(0, duration, int(fs * duration))

n_mics = 7
d = 0.1  # distancia entre micrófonos (en línea recta)

# Genero señales con retardos simulados para un ángulo de llegada dado
theta_deg = 60
theta = np.deg2rad(theta_deg)

sample_delays = []

for i in range(n_mics):
    d_i = i * d                            # Distancia del micrófono i al micrófono de referencia (mic_0)
    delay_seg = (d_i * np.cos(theta)) / c  # Retardo en segundos 
    delay_samples = int(delay_seg * fs)          
    sample_delays.append(delay_samples)

# Genero señales--------------------------------------------------------
pulse = np.zeros_like(t)
pulse[int(len(t)/2)] = 1.0

mic_signals = []

for delay in sample_delays:
    señal_retardada = np.roll(pulse, delay)  # Desplaza la señal 'pulse' en el tiempo simulando el retardo
    mic_signals.append(señal_retardada)      # Guarda la señal simulada en la lista



# Agrego ruido a las señales
mic_signals_rir = af.apply_reverb_synth(mic_signals, fs=fs, duration=duration, tau=7.23e-3, rir_A=0.18, p_noise = 0.02)


# Cálculos con las señales obtenidos-------------------------------------

# Calculo los TDOA respecto al primer micrófono // Se puede probar con CC o GCC (get_tau / get_taus_gcc_phat)
# Aca probar entre mic_signals y mic_signals_rir para ver diferencias
tau_list = af.get_taus_n_mic(mic_signals_rir, fs) 

# Calculo los diferentes angulos respecto a los tau anteriores
est_theta_list = af.get_direction_n_signals(d,tau_list, c, fs)

# Calculo el angulo promedio

theta_prom = (np.sum(est_theta_list[1:])) / (len(est_theta_list ) - 1)     # Promedio (no considero el inicial de referencia)

# Muestro los resultados------------------------------------------------
for i, (tau, est_theta) in enumerate(zip(tau_list, est_theta_list)):            # Mic 1 con valores nulos porque no hay TDOA para comparar 
    print(f"Mic {i+1}: TDOA = {tau:.6f} s, Ángulo estimado = {est_theta:.2f}°")

print(f"El ángulo promedio es: {theta_prom:.2f}°")

# Graficos
plot_dicts = []

# Armo un diccionario de cada uno para plotear
for i, signal in enumerate(mic_signals):
    color = f"C{i}"  
    dic = {
        "time vector": t,
        "signal": signal,
        "label": f"Micrófono {i+1}",
        "color": color
    }
    plot_dicts.append(dic)

# Plot 
plot.plot_signal(*plot_dicts, title=f"Señales en micrófonos (θ = {est_theta:.2f}°)", 
                 grid=True, legend=True, figsize=(10, 4), xlimits=(0.048, 0.052))

#------------------------TEMP

# Hago una IR para graficar y comparar solamente
rir_synth = af.rir(tau=7.23e-3, fs=fs, duration=duration)

import matplotlib.pyplot as plt

#plt.plot(t, mic_signals[0], label="Señal limpia")
plt.plot(t, mic_signals_rir[0], label="Señal con reverberación", alpha=0.7)
#plt.plot(t, leaking_noise, label="ruido")
plt.legend()
plt.grid(True)
plt.title("Señal resultante")
plt.show()
