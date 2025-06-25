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


def get_direction(d, tau, c=340, fs=44100):
    """
    Returns direction of arrival between 2 microphones
    Input:
        - d: float type object. Distance between microphones.
        - t: float type object. Time arrival difference between microphones.
        - c: Int type object. Sound propagation speed.
        - fs: Int type object. Sample Frequency.
    """
    t = tau/fs
    angle = np.arccos(c*t/d)
    angle = np.rad2deg(angle)
    return angle

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
                d_mics = d*((mic_num_2)-mic_num_1)
                print(f"Tau entre mic {mic_num_1} y {mic_num_2}")
                print(d_mics)
                max_tau = np.abs(d_mics/c)*fs
                tau = get_tau_V2(mic_signal_1, mic_signal_2, max_tau, mode=method)
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
