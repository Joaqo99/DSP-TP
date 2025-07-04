import soundfile as sf
from IPython.display import Audio
import numpy as np
from scipy import signal
from scipy import fft as scfft
import numpy as np
import colorednoise as cn
import filters
import json
import pyroomacoustics as pra
import pandas as pd

def cross_corr(x1, x2, fs=48000, mode="Classic"):
    '''
    Computes correlation between signals. 
    The correlation method varies with mode parameter if clasic cross correlation or generalized version with wightings.

    Input:
        - x1: array type object. Microphone 1 signal.
        - x2: array type object. Microphone 2 signal.
        - fs: int type object. Sample frequency.
        - mode: Str type object. Possible options:
            - "Classic". By default. Performs Classic Cross Correlation.
            - "ROTH". Performs Generalized Cross Correlation with ROTH wighting.
            - "SCOT". Performs Generalized Cross Correlation with SCOT wighting.
            - "PHAT". Performs Generalized Cross Correlation with PHAT wighting.
            - "ECKART". Performs Generalized Cross Correlation with Eckart wighting. This filter version is an approximation where the attenuation factor is neglected.
            - "HT". Performs Generalized Cross Correlation with HT wighting. ML weighting is considered the same as HT.
    Output:
        - corr: Array type object. Correlation output vector.
    '''

    def cs_ifft(x):
        """Performs ifft for weighted cross spectrum"""
        corr = get_ifft(x, input="complex", real = False)
        corr = np.fft.fftshift(corr)
        corr = np.real(corr)
        corr = np.roll(corr, -1)
        return corr

    #check lenght and executes zero pad if needed
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    
    N = len(x1) + len(x2) - 1  # N = 5

    # Zero padding
    x1_padded = np.pad(x1, (0, N - len(x1)))
    x2_padded = np.pad(x2, (0, N - len(x2)))

    #get fft
    freqs, x1_fft = get_fft(x1_padded, fs, normalize=False, nfft=N, output="complex", real_fft=False)
    freqs, x2_fft = get_fft(x2_padded, fs, normalize=False, nfft=N, output="complex", real_fft=False)

    cross_spect = x1_fft*np.conjugate(x2_fft)

    #weightings
    if isinstance(mode, str):
        if mode == "Classic":
            psi = 1
        elif mode == "ROTH":
            psi = filters.roth(x1_fft)
        elif mode == "SCOT":
            psi = filters.scot(x1_fft, x2_fft)
        elif mode == "PHAT":
            psi = filters.phat(cross_spect)
        elif mode == "ECKART":
            psi = filters.eckart(x1_fft, x2_fft, x1, x2)
        elif mode == "HT":
            psi = filters.ht(x1_fft, x2_fft, cross_spect)
        else:
            raise ValueError('mode parameter must be either "Classic", "Roth", "Scot" or "PHAT".')
        weighted_cs = psi*cross_spect
        corr = cs_ifft(weighted_cs)
        return corr
    else:
        raise ValueError('mode parameter must be a String object and either "Classic", "Roth", "Scot" or "PHAT".')


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
    corr_full = cross_corr(mic_2, mic_1, mode=mode)
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

def get_taus_n_mic(mic_signals, d, fs=48000, mode="Classic"):
    """
    Calcula el tiempo de arribo relativo (TDOA) entre el primer micrófono y los demás.

    Input:
        - mic_signals: lista de arrays. Cada array es la señal de un micrófono.
        - fs: frecuencia de muestreo

    Output:
        - taus: lista de retardos en segundos. El primer micrófono tiene retardo 0.
    """
    reference = mic_signals[0]
    taus = [0.0]  # El micrófono de referencia tiene retardo 0

    for i in range(1, len(mic_signals)):
        tau = get_tau(reference, mic_signals[i], d=d, fs=fs, mode=mode)
        taus.append(tau)
    
    return taus

def get_direction(d, t, c=340, fs=48000):
    """
    Returns direction of arrival between 2 microphones
    Input:
        - d: float type object. Distance between microphones.
        - t: float type object. Time arrival difference between microphones.
        - c: Int type object. Sound propagation speed.
        - fs: Int type object. Sample Frequency.
    """
    arg = (t * c) / d
    arg = np.clip(arg, -1.0, 1.0)  # limitar argumento entre -1 y 1
    angle = np.arccos(arg)
    angle = np.rad2deg(angle)
    return angle

def get_direction_n_signals(d,taus, c=343, fs=48000):
    """
    Calcula los ángulos estimados de llegada del sonido para múltiples micrófonos.
    
    Input:
        - taus: list of float. Lista de retardos respecto al micrófono de referencia.
        - d: float. Distancia entre micrófonos.
        - c: float. Velocidad del sonido (por defecto 343 m/s).
        - fs: int. Frecuencia de muestreo (por defecto 48000 Hz).
    
    Output:
        - angles: list of float. Ángulos estimados en grados.
    """    
    angles = []
    for i, tau in enumerate(taus):
        if i == 0:
            angles.append(0.0)  # Micrófono de referencia
        else:
            d_total = d * i
            angle = get_direction(d_total, tau, c=c, fs=fs)
            angles.append(angle)
    return angles


def conv(in_signal, ir):
    """Performs convolution"""
    return signal.fftconvolve(in_signal, ir, mode='same')

def moving_media_filter(in_signal, N):
    """Performs media moving filter"""
    ir = np.ones(N) * 1 / N
    return conv(in_signal, ir)

def load_audio(file_name):
    """
    Loads a mono or stereo audio file in audios folder.
    Input:
        - file_name: String type object. The file must be an audio file.
    Output:
        - audio: array type object.
        - fs: sample frequency
        - prints if audio is mono or stereo.
    """
    if type(file_name) != str:
        raise Exception("file_name must be a string")

    audio, fs = sf.read(f"{file_name}")

    return audio , fs

def save_audio(file_name, audio, fs=48000):
    """
    Save an audio signal to a file in WAV format.

    Parameters:
        - file_name (str): Name of the output WAV file.
        - audio (ndarray): Audio signal to save.
        - fs (int, optional): Sampling rate. Default is 48000.

    Returns:
        None
    """
    if type(file_name) != str:
        raise Exception("file_name must be a string")

    sf.write(file_name, audio, fs)

    return 

def get_audio_time_array(audio, fs):
    """
    Returns audio time array
    Input:
        - audio: array type object.
        - fs: Int type object. Sample rate.
    Output:
        - duration: int type object. Audio duration
        - time_array: array type object.
    """
    #error handling
    if  type(audio) != np.ndarray:
        raise ValueError("audio must be a ndarray")
    if type(fs) != int:
        raise ValueError("fs must be int")
    
    #features
    duration = len(audio) / fs
    time_array = np.linspace(0, duration, len(audio))

    return duration, time_array

def play_audio(audio, fs):
    """
    Plays a mono audio
    Inputs:
        - audio: array type object. Audio to play. Must be mono.
        - fs: int type object. Sample rate
    """
    #error handling
    if type(fs) != int:
        raise ValueError("fs must be int")

    return Audio(audio, rate=fs)

def to_mono(audio):
    """
    Converts a stereo audio vector to mono.
    Insert:
        - audio: array type object of 2 rows. Audio to convert.
    Output:
        - audio_mono: audio converted
    """
    #error handling
    if  type(audio) != np.ndarray:
        raise ValueError("audio must be a ndarray")
    if len(audio.shape) == 1:
        raise Exception("Audio is already mono")
    elif audio.shape[0] != 2 and audio.shape[1] != 2: 
        raise Exception("Non valid vector")
    
    #features
    audio_mono = (audio[:,0]/2)+(audio[:,1]/2)
    return audio_mono

def reverb(ir, audio, ir_fs, audio_fs):
    """
    Returns an auralization of an audio and a given impulse response
    Input:
        - ir: array type object. Impulse response
        - audio: array type object. Must be mono audio.
        - ir_fs: int type object. Impulse response sample rate.
        - audio_fs: int type object. Audio sample rate.
    Output:
        - audio_auralized: array type object.
    """

    #error handling
    if type(ir) != np.ndarray or type(audio) != np.ndarray:
        raise ValueError("both audio and ir must be a ndarray")
    if type(ir_fs) != int or type(audio_fs) != int:
        raise ValueError("fs must be int")
    
    assert ir_fs == audio_fs, "Both Impulse Response and Audio sample rates must be the same"
    
    if len(audio.shape) != 1:
        raise Exception("Audio must be mono")
    
    #features
    audio_auralized = signal.fftconvolve(ir, audio)
    return audio_auralized

def to_dB(audio):
    """
    Returns an audio amplitude array in dB scale
    Input:
        - audio: array type object.
    Output:
        - audio_db: array type object.
    """
    if  type(audio) != np.ndarray:
        raise ValueError("audio must be a ndarray")
    
    audio_db = 10*np.log10(audio**2)
    return audio_db

def get_fft(in_signal, fs, normalize=True, output="mag-phase", real_fft=True, nfft = None):
    """
    Performs a fast fourier transform over the input signal. As we're working with real signals, we perform the rfft.
    Input:
        - in_signal: array or type object. input signal.
        - fs: int type object. Sample frequency
        - normalize: bool type object. If true, returns the normalized magnitude of the input signal. If output is "complex" it wont work
        - output: str type object. Output format, can be:
            - "mag-phase" for the magnitude and phase of the rfft. Default.
            - "complex" for the raw rfft.
        - real: bool type object. If true, returns rfft.

    If Output = mag_phase:
        - in_freqs: array type object. Real Frequencies domain vector.
        - fft_mag: array type object. Real Frequencies amplitude vector.
        - fft_phase: array type object. RealFrequencies phase vector.
    If Output = complex:
        - in_freqs: array type object. Real Frequencies domain vector.
        - fft: array type object. Real Frequencies raw fft vector.
    """
    if real_fft:
        rfft = scfft.rfft(in_signal, n = nfft)
        in_freqs = np.linspace(0, fs//2, len(rfft))
    else:
        rfft = np.fft.fft(in_signal, n = nfft)
        in_freqs = np.fft.fftfreq(len(in_signal), d=1/fs)

    #import pdb;pdb.set_trace()

    if output == "complex":
        return in_freqs, rfft
    elif output == "mag-phase":
        rfft_mag = abs(rfft)/len(rfft)
        if normalize: rfft_mag = rfft_mag / np.max(abs(rfft_mag))
        rfft_phase = np.angle(rfft)
        return in_freqs, rfft_mag, rfft_phase
    else:
        raise ValueError('No valid output format - Must be "mag-phase" or "complex"')

def get_ifft(in_rfft, in_phases=False, input="mag-phase", nfft = None, real = True):
    """
    Performs an inverse fast Fourier transform of a real signal
    Input:
        - in_rfft_mag: array type object. It must contain only the positive frequencies of the spectrum of the signal.
        - in_phases: array type object. It must contain only the positive frequencies of the spectrum of the signal. If false, it assumes the phases of all components are 0º.
        - input: str type object. Input format, can be "mag-phase" or "complex", "mag_phase" by default. If "complex", there must not be in_phases kwarg.
        - real: bool type object. If true, returns rifft.
    Output:
        - temp_signal: array type object. Transformed signal.
    """
    if input == "mag-phase":
        if type(in_phases) == bool and in_phases == False:
            in_phases = np.zeros(len(in_rfft))
    
        in_rfft = in_rfft * np.exp(1j * in_phases)
    elif input == "complex":
        if in_phases:
            raise Exception('If "complex" input there must not be a phase array input.')
    else:
        raise ValueError('Input format must be "mag_phase" or "complex"')
    
    if real:
        temp_signal = scfft.irfft(in_rfft, n = nfft)
    else:
        temp_signal = scfft.ifft(in_rfft, n = nfft)
    return temp_signal


def apply_noise(mic_signals, fs=48000, A_noise = 0.1, duration=0.1):
    """
    Applies pink noise to a list of microphone signals.

    Input:
        - mic_signals: List type object. List of microphone signals.
        - fs: int type object. Sample frequency.
        - A_noise: float type object. Pink noise amplitude.
        - duration: float type object. Total duration of each signal in seconds.

    Output:
        - mic_signals_rir : array type object. List of signals with added reverb.
    """
    mic_signals_rir = []
    
    for sig in mic_signals:
        # Ruido
        beta = 1  # 1 = ruido rosa, 0 = blanco, 2 = browniano
        # Generar ruido rosa
        A = A_noise
        pink_noise = A * cn.powerlaw_psd_gaussian(beta, int(fs*duration))
        #Sumo el pulso
        signal_rir = sig + pink_noise
        #Normalizo
        signal_rir = signal_rir / np.max(np.abs(signal_rir))
        mic_signals_rir.append(signal_rir)
      
    return mic_signals_rir

def angle_error(angle_true, angle_pred):
    """
    Performs a mean squared error between true angle of arrival and predicted angle of arrival. Angles must be both in degrees.
    """

    angle_true = np.deg2rad(angle_true)
    angle_pred = np.deg2rad(angle_pred)
    delta_angle = (angle_true - angle_pred + np.pi)%(2*np.pi) - np.pi
    mse = delta_angle**2/np.pi**2
    return mse

def gen_simulation_dict(name, *mods_dict, audio_filename="audios_anecoicos/delta.wav"):
    """
    Generates a dictionary with particular simulation conditions.
    Input:
        - name: str type object.
        - mods_dict: dictionary with modifications from default simulation. Contains:
            - var: room, source, mic_array.
            - param: selected parameteres from the variable to modify.
            - value: selected value for parameter.


    Parámetros de cada simulación:
    - Room: Dimensiones, T60, relación señal ruido. Nombres de las variables: (room: {dim, t60, snr, reflections_order})
    - Fuente: Posición/ángulo de la fuente. Señal. Nombres de las variables: (source: {position, signal, fs})
    - Array: cantidad de micrófonos, distancia entre micrófonos, patrón polar. Nombres de las variables: (mic_array: {n, d, pol_pat, position})
        
    """

    def default_simulation():
        """
        Generates a default simulation dictionary.
        """

        fs = 48000


        default = {
            "name":"dafult",
            "room" : {"dim":[5, 6, 2], "t60":1, "snr":40, "reflex_order":100},
            "source": {"position":[1, 1, 1], "audio_filename":audio_filename, "fs":48000},
            "mic_array": {"n":4, "d": 0.1, "pol_pat": "omni", "position":[5, 5, 1]},
        }

        return default

    new_dict = default_simulation()
    new_dict["name"] = name
    for mod in mods_dict:
        var = mod["var"]
        param = mod["param"]
        value = mod["value"]
        new_dict[var][param] = value

    
    filename = f"simulaciones/{name}"
    with open(filename, "w") as f:
        json.dump(new_dict, f, indent=4) # indent for pretty-printing (optional)
        print(f"Simulación generada en {filename}")



def simulate(sim_config_name):
    try:
        # 1) Levantamos los datos de la configuración de simulación
        with open(f"simulaciones/{sim_config_name}", "r") as f:
            sim_config = json.load(f)

        # 2) Generar room con pyroom
        room_dim = sim_config["room"]["dim"]
        rt60 = sim_config["room"]["t60"]

        room_dim_x, room_dim_y, room_dim_z = room_dim 
        eabs, _ = pra.inverse_sabine(rt60, room_dim)

        fs = sim_config["source"]["fs"]
        snr = sim_config["room"]["snr"]
        reflex_order = sim_config["room"]["reflex_order"]

        temperature = 20
        humidity = 40

        room = pra.ShoeBox(
            room_dim, fs=fs, max_order=reflex_order,
            materials=pra.Material(eabs),
            temperature=temperature, humidity=humidity,
            air_absorption=True
        )

        # 3) Coloco micrófonos
        d_mic = sim_config["mic_array"]["d"]
        n_mic = sim_config["mic_array"]["n"]
        mic_array_pos = sim_config["mic_array"]["position"]

        mics_pos = []

        x, y, z = mic_array_pos
        if y > room_dim_y or y < 0:
            raise ValueError(f"Mic array out of Y bounds en {sim_config_name}")
        elif z > room_dim_z or z < 0:
            raise ValueError(f"Mic array out of Z bounds en {sim_config_name}")

        for n in range(n_mic):
            new_x = x - d_mic * n
            if new_x < 0 or new_x > room_dim_x:
                raise ValueError(f"Mic array out of X bounds en {sim_config_name}")
            loc = [new_x, y, z]
            mics_pos.append(loc)

        mic_array_loc = np.c_[*mics_pos]
        room.add_microphone_array(mic_array_loc)

        # 4) Coloco fuente
        source_pos = sim_config["source"]["position"]
        audio_filename = sim_config["source"]["audio_filename"]
        audio, _ = load_audio(audio_filename)
        room.add_source(source_pos, signal=audio)

        # 5) Simulo
        room.simulate(snr=snr)
        return room

    except Exception as e:
        print(f"Falló la simulación '{sim_config_name}': {e}")
        return None


def process_simulation_data(*sim_configs, c=343, df_values=False):
    """
    Procesa múltiples simulaciones. Devuelve un DataFrame en formato largo con:
    - expected_theta
    - método de estimación
    - ángulo estimado promedio (theta_prom)
    - error cuadrático medio (error)
    - lista de ángulos estimados (est_theta_list)

    Si df_values es una lista como ["room.snr", "mic_array.d"], también incluye esos valores con ese nombre de columna.
    """
    methods = ["Classic", "ROTH", "PHAT", "SCOT", "ECKART", "HT"]
    rows = []

    for sim_conf_name in sim_configs:
        with open(f"simulaciones/{sim_conf_name}", "r") as f:
            sim_conf = json.load(f)

        # Datos base
        source_pos = sim_conf["source"]["position"]
        array_pos = sim_conf["mic_array"]["position"]
        d = sim_conf["mic_array"]["d"]
        n = sim_conf["mic_array"]["n"]
        fs = sim_conf["source"]["fs"]

        # Centrar el array
        poss_mic_x, poss_mic_y, poss_mic_z = array_pos
        arr_center_x = poss_mic_x + (d * n) / 2
        source_x, source_y, source_z = source_pos

        
        # ----------Ángulo esperado en grados-----------

        expected_theta = np.arccos(np.abs((source_x - arr_center_x)) / 
                    (np.sqrt((source_x - arr_center_x)**2 + (source_y - poss_mic_y)**2 + (source_z - poss_mic_z)**2)))
        expected_theta = np.round(np.rad2deg(expected_theta), 3)   
        if source_x < poss_mic_x:
           expected_theta = 180 - expected_theta
        

        # Simulación
        room = simulate(sim_conf_name)
        mic_signals = room.mic_array.signals

        sos_filter = filters.anti_alias_filter(c, d, fs, order=1)
        mic_signals = [signal.sosfilt(sos_filter, x) for x in mic_signals]

        for method in methods:
            try:
                tau_list = get_taus_n_mic(mic_signals, fs, mode=method)
                est_theta_list = get_direction_n_signals(d, tau_list, c, fs)
                est_theta_list = [np.round(x, 2) for x in est_theta_list]

                theta_prom = np.mean(est_theta_list[1:])
                error = angle_error(expected_theta, theta_prom)

                row = {
                    "sim_name": sim_conf_name,
                    "method": method,
                    "expected_theta": expected_theta,
                    "theta_prom": round(theta_prom, 3),
                    "error": round(error, 5),
                    "est_theta_list": est_theta_list
                }

                # Agregar valores extras con nombres como "mic_array.d"
                if isinstance(df_values, list):
                    for full_key in df_values:
                        if '.' in full_key:
                            section, key = full_key.split('.', 1)
                            value = sim_conf.get(section, {}).get(key, None)
                            row[full_key] = value

                rows.append(row)

            except Exception as e:
                print(f"Error en {sim_conf_name} con método {method}: {e}")
                continue

    return pd.DataFrame(rows)