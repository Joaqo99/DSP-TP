import numpy as np
import plot
import audio_functions as af
from scipy.signal import correlate



c = 343    
fs=44100
duration = 0.01      # 1 seg

t = np.linspace(0, duration, int(fs * duration))



# Simulación de 3 micrófonos
n_mics = 6
d = 0.1  # distancia entre micrófonos (en línea recta)

# Generamos señales con retardos simulados para un ángulo de llegada dado
theta_deg = 60
theta = np.deg2rad(theta_deg)

sample_delays = []

for i in range(n_mics):
    d_i = i * d                            # Distancia del micrófono i al micrófono de referencia (mic_0)
    delay_seg = (d_i * np.cos(theta)) / c  # Retardo en segundos 
    delay_samples = int(delay_seg * fs)          
    sample_delays.append(delay_samples)

# Generar señales
pulse = np.zeros_like(t)
pulse[int(len(t)/2)] = 1.0

mic_signals = []

for delay in sample_delays:
    señal_retardada = np.roll(pulse, delay)  # Desplaza la señal 'pulse' en el tiempo simulando el retardo
    mic_signals.append(señal_retardada)      # Guarda la señal simulada en la lista

# Calcular los TDOA respecto al primer micrófono
tau_list = af.get_taus_n_mic(mic_signals, fs)        

est_theta_list = []

# Calcular ángulos estimados
for i, tau in enumerate(tau_list):
    if i == 0:
        est_theta_list.append(0)  # El micrófono de referencia no tiene ángulo (por definición)
    else:
        d_total = d * i  # Distancia entre mic_0 y mic_i
        angulo = af.get_direction(d_total, tau, c=343, fs=fs)
        est_theta_list.append(angulo)

# Mostrar resultados
for i, (tau, est_theta) in enumerate(zip(tau_list, est_theta_list)):
    print(f"Mic {i+1}: TDOA = {tau:.6f} s, Ángulo estimado = {est_theta:.2f}°")

# Grafico
plot_dicts = []

for i, signal in enumerate(mic_signals):
    color = f"C{i}"  # Usará colores estándar de matplotlib: C0, C1, C2...
    dic = {
        "time vector": t,
        "signal": signal,
        "label": f"Micrófono {i+1}",
        "color": color
    }
    plot_dicts.append(dic)

# Llamás a la función pasando los diccionarios como argumentos individuales
plot.plot_signal(*plot_dicts, title=f"Señales en micrófonos (θ = {est_theta:.2f}°)", 
                 grid=True, legend=True, figsize=(10, 4), xlimits=(0.004, 0.008))

