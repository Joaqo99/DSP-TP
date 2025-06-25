import numpy as np
import plot
import audio_functions as auf
import filters
from scipy import signal
import pandas as pd
import json
import time

def get_tau_V2(mic_1, mic_2, max_tau, mode="Classic"):
    """
    Gets the arrival time diference between 2 microphones
    Input:
        - mic_1: array type object. Microhpone 1 signal.
        - mic_2: array type object. Microhpone 2 signal.
        - fs: 
    Output:
        t: float type object. Arrival time diference
    """
    corr = auf.cross_corr(mic_2, mic_1, mode=mode)
    n_corr = np.arange(-len(mic_2) +1, len(mic_1))
    corr = corr[int(len(corr)/2 - max_tau) : int(len(corr)/2 + max_tau)]
    n_corr = n_corr[int(len(n_corr)/2 - max_tau) : int(len(n_corr)/2 + max_tau)]
    tau = (n_corr[np.argmax(corr)])
    return tau

def get_tau(mic_1, mic_2, d, fs=48000, c=343, mode="Classic"):
    """
    Gets the arrival time difference (TDOA) between 2 microphones,
    limiting search to max physically possible tau based on mic spacing.
    
    Input:
        - mic_1 (np.ndarray): Signal from the first microphone.
        - mic_2 (np.ndarray): Signal from the second microphone.
        - d (float): Distance between the microphones in meters.
        - fs (int, optional): Sampling rate in Hz. Default is 48000.
        - c (float, optional): Speed of sound in m/s. Default is 343.
        - mode (str, optional): Correlation mode used in `cross_corr`. 
                              Typically "Classic" or another implemented method.
    Returns:
        - float: Estimated TDOA (time delay) in seconds.
    """
    tau_max_sec = d / c
    tau_max_samples = int(np.round(tau_max_sec * fs))
       
    # Correlación completa y vector de lags completo
    corr_full = auf.cross_corr(mic_2, mic_1, mode=mode)
    n_full = np.arange(-len(mic_2)+1, len(mic_1))
    
    mid = len(n_full)//2
    
    # Ventana [mid-tau_max : mid+tau_max]
    A = 5 # Escalar para agrandar la ventana
    start = mid - tau_max_samples * A
    end   = mid + tau_max_samples * A + 1   # +1 porque el corte de Python no incluye el extremo
    
    # Clamp para no salir de índices
    start = max(start, 0)
    end   = min(end, len(n_full))
    
    n_cut  = n_full[start:end]
    corr_cut = corr_full[start:end]
    
    idx_local = np.argmax(corr_cut)
    lag = n_cut[idx_local]

    tau = lag / fs

    return tau

def get_direction(d, tau, c=343, fs =44100):
    """
    d: distancia entre micrófonos (m)
    tau: TDOA en segundos (ya calculado como muestras/fs)
    c: velocidad sonido (m/s)
    """
    arg = c * tau / d
    arg = np.clip(arg, -1.0, 1.0)
    angle = np.arccos(arg)
    return np.rad2deg(angle)

methods = ["Classic", "ROTH", "PHAT", "SCOT", "ECKART", "HT"]

def doa_system(mic_signals_list, d, fs, c=343, method="Classic"):
    sos_filter = filters.anti_alias_filter(c, d, fs, order=1)
    mic_signals_list = [signal.sosfilt(sos_filter, x) for x in mic_signals_list]

    tau_matrix = []
    theta_matrix = []
    for mic_num_1, mic_signal_1 in enumerate(mic_signals_list):
        for mic_num_2, mic_signal_2 in enumerate(mic_signals_list):
            mic_tau_list = []
            mic_theta_list = []

            if mic_num_1 == mic_num_2:
                mic_tau_list.append(0.0)
                mic_theta_list.append(0.0)
            else:
                #d_mics = d*((mic_num_2)-mic_num_1)
                delta = mic_num_2 - mic_num_1
                d_mics = abs(delta) * d
                tau = get_tau(mic_signal_1, mic_signal_2, d, mode=method)
                mic_tau_list.append(tau)
                theta = get_direction(d_mics, tau, c=c, fs=fs)
                mic_theta_list.append(theta)

        tau_matrix.append(mic_tau_list)
        theta_matrix.append(mic_theta_list)

    tau_matrix = np.array(tau_matrix)
    theta_matrix = np.array(theta_matrix)
    print(theta_matrix)
    theta_matrix = np.round(theta_matrix, 2)
    theta_matrix_flattened = theta_matrix.flatten()
    theta_matrix_cleaned = np.delete(theta_matrix_flattened, np.where(theta_matrix_flattened == 0.0))
    theta_prom = np.mean(theta_matrix_cleaned)

    return theta_prom, theta_matrix


def process_simulation_data(*sim_configs, c=343):
    """
    Procesa múltiples simulaciones. Devuelve un DataFrame en formato largo con:
    - expected_theta
    - método de estimación
    - ángulo estimado promedio (theta_prom)
    - error cuadrático medio (error)
    - lista de ángulos estimados (est_theta_list)

    Cada fila del DataFrame representa una simulación y un método.
    """

    methods = ["Classic", "ROTH", "PHAT", "SCOT", "ECKART", "HT"]  # Reemplazá por los métodos que estés usando


    rows = []

    for sim_conf_name in sim_configs:
        # --- Cargar configuración de simulación ---
        with open(f"simulaciones/{sim_conf_name}", "r") as f:
            sim_conf = json.load(f)

        # --- Parámetros básicos ---
        source_pos = sim_conf["source"]["position"]
        array_pos = sim_conf["mic_array"]["position"]  # referencia del array
        d = sim_conf["mic_array"]["d"]
        n = sim_conf["mic_array"]["n"]
        fs = sim_conf["source"]["fs"]

        # Centrar el array
        arr_center_x, arr_center_y, arr_center_z = array_pos
        arr_center_y += (d * n) / 2
        source_x, source_y, source_z = source_pos

        # Ángulo esperado en grados
        expected_theta = np.arccos(np.abs(source_x - arr_center_x) / np.sqrt(
            (source_x - arr_center_x)**2 + (source_y - arr_center_y)**2 + (source_z - arr_center_z)**2
        ))
        expected_theta = np.round(np.rad2deg(expected_theta), 3)

        # --- Simulación ---
        room = auf.simulate(sim_conf_name)
        mic_signals = room.mic_array.signals

        for m in methods:
            theta_prom, theta_matrix = doa_system(mic_signals, d, fs, method=m)
            error = np.mean((expected_theta - theta_prom) ** 2)
            rows.append({
                "sim_name": sim_conf_name,
                "expected_theta": expected_theta,
                "method": m,
                "theta_prom": round(theta_prom, 3),
                "error": round(error, 4),
                "est_theta_list": theta_matrix
            })


    return pd.DataFrame(rows)




sim_names = []

for i in range(1):
    mod_dict = {"var":"room", "param":"t60", "value":(0.4+i*0.15)}
    #mod_dict = {"var":"room", "param":"dim", "value":[7,8, 3]}
    #mod_dict = {"var":"mic_array", "param":"position", "value":[3,3, 1.5]}
    sim_name = f"prueba t60-{0.4+i*0.15}"
    auf.gen_simulation_dict(sim_name, mod_dict)
    sim_names.append(sim_name)
    time.sleep(1)

df = process_simulation_data(*sim_names)
print(df)
