import numpy as np, time
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

def get_tau(mic_1, mic_2, d, fs=48000, c=343, mode="Classic", A = 1):
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
    #corr_full = signal.correlate(mic_2, mic_1, mode="full")
    corr_full = auf.cross_corr(mic_2, mic_1, mode=mode)
    
    n_full = np.arange( - len(mic_2) + 1, len(mic_1))

    mid = len(n_full) // 2
    
    # Ventana [mid-tau_max : mid+tau_max]
    # A: Escalar para agrandar la ventana
    start = mid - tau_max_samples * A
    end   = mid + tau_max_samples * A + 1   # +1 porque el corte de Python no incluye el extremo
    
    # Clamp para no salir de índices
    start = max(start, 0)
    end   = min(end, len(n_full))
    
    n_cut  = n_full[start:end]
    corr_cut = corr_full[start:end]
    
    lag = n_cut[np.argmax(corr_cut)] # cantidad de muestras desplazadas entre una señal y la otra

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
    print(f"Calculo angulos por método: {method} \n")
    tau_matrix = []
    theta_matrix = []
    for i in range(len(mic_signals_list)):
        mic_theta_list = []
        for j in range(len(mic_signals_list)):
            if i == j:
                mic_theta_list.append(0.0)
            elif j > i:
                # solo calcular una vez por par
                d_mics = abs(j - i) * d
                tau = get_tau(mic_signals_list[i], mic_signals_list[j], d_mics, fs=fs, mode=method, A = 5)
                theta = get_direction(d_mics, tau, c=c)
                mic_theta_list.append(theta)
                print(f"Theta de {j}-{i}: {np.round(theta, 3)}")
            else:
                mic_theta_list.append(None)  # o np.nan si preferís
            
        theta_matrix.append(mic_theta_list)

    tau_matrix = np.array(tau_matrix)
    theta_matrix = np.array(theta_matrix, dtype=float)

    if theta_matrix is not None:
        theta_matrix = np.round(theta_matrix, 2)

    # aplanamos
    theta_matrix_flattened = np.array(theta_matrix).flatten()
    # filtramos: nos quedamos sólo con valores que NO sean nan y NO sean 0
    mask = (~np.isnan(theta_matrix_flattened)) & (theta_matrix_flattened != 0)
    # aplicamos la máscara
    theta_matrix_cleaned = theta_matrix_flattened[mask]
    # promedio ya sin nan ni ceros
    theta_prom = np.mean(theta_matrix_cleaned)

    print(f"\nTheta matrix {method}: \n {theta_matrix}\n")
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

        # ----------Ángulo esperado en grados-----------
        
        expected_theta = np.arccos(np.abs(source_x - arr_center_x) / np.sqrt(
            (source_x - arr_center_x)**2 + (source_y - arr_center_y)**2 
        ))                                                        #        + (source_z - arr_center_z)**2
        expected_theta = np.round(np.rad2deg(expected_theta), 3)
        
        """
        
        #  OTRO METODO

        dx = source_x - arr_center_x
        dy = source_y - arr_center_y

        # Distancia en el plano XY
        r_xy = np.hypot(dx, dy)
        if r_xy == 0:
            expected_theta = 0.0
        else:
            # Ángulo medido desde el eje Y (baseline) hacia la dirección de llegada
            # Si tu baseline es vertical (eje Y), cos(theta) = dy / r_xy
            expected_theta = np.rad2deg(np.arccos(dy / r_xy))
        expected_theta = np.abs(180 - np.round(expected_theta, 3))
        """


        print(f"SIMULACION: {sim_conf_name}")
        print(f"Expected theta: {np.round(expected_theta,2)}")
        print(f"Distancia total: {np.round(np.sqrt((source_x - arr_center_x)**2 + (source_y - arr_center_y)**2 + (source_z - arr_center_z)**2),2)}")
        print(f"Diferencia eje X: {np.round(abs(source_x - arr_center_x),2)}")
        print(f"Diferencia eje Y: {np.round(abs(source_y - arr_center_y),2)}")
        
        theta_exp_rad = np.deg2rad(expected_theta)
        tau01_theo = (d * np.cos(theta_exp_rad)) / c
        print(f"τ teórico (mic0→mic1): {np.round(tau01_theo,8)}")


        # --- Simulación ---
        room = auf.simulate(sim_conf_name)
        #if room is None:
        #    continue  # saltar esta simulación y seguir con la siguiente
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

def generate_batch_simulations(base_name, t60, n_config=100):
    sim_names = []
    # Valores fijos de sala (tú puedes variarlos también si quieres)
    dim = [5, 5, 2]
    for i in range(n_config):
        sim_name = f"{base_name}_T60_{t60:.1f}_run{i + 1}"
        # Modificación de RT60
        room_mod = {"var":"room", "param":"t60", "value":t60}

        # Mic array: base en el centro de X–Y, z fija
        # Queremos que queden dentro de [0, dim]
        base_mic_pos = np.array([dim[0]/2, dim[1]/2, 1.0])
        mic_pos = base_mic_pos + np.random.uniform(-0.5, 0.5, size=3)
        # Clamp para que quede en [0,dim]
        mic_pos[0] = np.clip(mic_pos[0], 0.1, dim[0]-0.1)
        mic_pos[1] = np.clip(mic_pos[1], 0.1, dim[1]-0.1)
        mic_pos[2] = np.clip(mic_pos[2], 0.5, dim[2]-0.5)
        mic_mod = {"var":"mic_array", "param":"position", "value":mic_pos.tolist()}

        # Fuente: posición aleatoria dentro de [0.5, dim-0.5]
        src_pos = [
            float(np.random.uniform(0.5, dim[0]-0.5)),
            float(np.random.uniform(0.5, dim[1]-0.5)),
            float(np.random.uniform(0.5, dim[2]-0.5))
        ]
        src_mod = {"var":"source", "param":"position", "value":src_pos}

        # Generar diccionario
        auf.gen_simulation_dict(sim_name,
                               room_mod,
                               mic_mod,
                               src_mod)
        sim_names.append(sim_name)
        time.sleep(0.01)
    return sim_names

t60_list = [0.2, 0.4, 0.6, 0.8]
"""
# 1) Generar todos los nombres de simulaciones
t60_list = [0.2, 0.4, 0.6, 0.8]
all_sim_names = []
for t60 in t60_list:
    sim_names = generate_batch_simulations("batch", t60, n_config=100)
    all_sim_names.extend(sim_names)

# 2) Procesar TODAS las simulaciones de una vez
df = process_simulation_data(*all_sim_names)

# 3) Guardar y mostrar el resultado final solo una vez
df.to_csv("resultados_batch.csv", index=False)
print(df)
"""
all_dfs = []
for t60 in t60_list:
    sim_names = generate_batch_simulations("sim", t60, n_config=5)
    df_t60 = process_simulation_data(*sim_names)
    df_t60["t60"] = t60
    all_dfs.append(df_t60)
    print(f"--- Resultados para T60 = {t60} ---")
    print(df_t60)

# Y si quieres un solo DataFrame final con todo:
df_all = pd.concat(all_dfs, ignore_index=True)
df_all.to_csv("resultados_todos.csv", index=False)

