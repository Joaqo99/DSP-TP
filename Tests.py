import numpy as np
import plot
from scipy.signal import correlate
import audio_functions as af
c = 343              
d = 0.1              # Distancia entre mics

theta_deg = 60       
theta = np.deg2rad(theta_deg)

fs = 44100          
duration = 0.01      # 1 seg
t = np.linspace(0, duration, int(fs * duration))

# Simulación de señal (pulso)
pulse = np.zeros_like(t)
pulse[int(len(t)/2)] = 1.0  

sample_delay = int((d * np.cos(theta)) / c * fs)

# Señales en los micrófonos
y1 = pulse
y2 = np.roll(pulse, sample_delay)  # señal retardada

# Cálculo del retardo
tau = af.get_tau(y1,y2)

print(f"TDOA = {tau:.6f} s ({sample_delay} muestras)")

est_theta = af.get_direction(d, tau, c=343)


print(f"Estimado TDOA: {tau:.6f} s")
print(f"Ángulo estimado: {est_theta:.2f}°")

mic1_dict = {"time vector": t,"signal": y1,"label": "Micrófono 1","color": "blue"}

mic2_dict = {"time vector": t,"signal": y2,"label": "Micrófono 2","color": "red"}

plot.plot_signal(mic1_dict, mic2_dict,title=f"Señales en micrófonos (θ = {est_theta}°)", grid=True, legend=True, figsize=(10, 4),
            xlimits=(0.004, 0.006))  # Ajustá según dónde cae el pulso)


