import numpy as np
import plot
import audio_functions as af
from scipy.signal import correlate


c = 343    
fs=44100
duration = 0.01      # 1 seg

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

# Genero señales
pulse = np.zeros_like(t)
pulse[int(len(t)/2)] = 1.0

mic_signals = []

for delay in sample_delays:
    señal_retardada = np.roll(pulse, delay)  # Desplaza la señal 'pulse' en el tiempo simulando el retardo
    mic_signals.append(señal_retardada)      # Guarda la señal simulada en la lista



# Agrego ruido a las señales
mic_signals_rir = af.apply_reverb_synth(mic_signals, fs=fs, phi = -60, duration=duration)



# Calculo los TDOA respecto al primer micrófono // Se puede probar con CC o GCC (get_tau / get_taus_gcc_phat)
tau_list = af.get_taus_gcc_phat_n_mic(mic_signals, fs) 

# Calculo los diferentes angulos respecto a los tau anteriores
est_theta_list = af.get_direction_n_signals(d,tau_list, c, fs)

# Calculo el angulo promedio

theta_prom = (np.sum(est_theta_list[1:])) / (len(est_theta_list ) - 1)     # Promedio (no considero el inicial de referencia)

# Muestro los resultados
for i, (tau, est_theta) in enumerate(zip(tau_list, est_theta_list)):            # Mic 1 con valores nulos porque no hay TDOA para comparar 
    print(f"Mic {i+1}: TDOA = {tau:.6f} s, Ángulo estimado = {est_theta:.2f}°")

print(f"El ángulo promedio es: {theta_prom:.2f}°")

# Grafico
plot_dicts = []

# Armo un diccionario por cada uno para plotear
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
                 grid=True, legend=True, figsize=(10, 4), xlimits=(0.004, 0.008))

