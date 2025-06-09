import soundfile as sf
from IPython.display import Audio
import numpy as np
from scipy import signal
from scipy import fft as scfft
import numpy as np
import colorednoise as cn
import filters

def cross_corr(x1, x2, fs=44100, mode="Classic"):
    '''
    Computes correlation between signals. 
    The correlation method varies with mode parameter if clasic cross correlation or generalized version with wightings.

    Input:
        - x1: array type object. Microphone 1 signal.
        - x2: array type object. Microphone 2 signal.
        - fs: int type object. Sample frequency.
        - mode: Str type object. Possible options:
            - "Classic". By default. Performs Classic Cross Correlation.
            - "Roth". By default. Performs Generalized Cross Correlation with Roth wighting.
            - "Scot". By default. Performs Generalized Cross Correlation with Scot wighting.
            - "PHAT". By default. Performs Generalized Cross Correlation with PHAT wighting.
    Output:
        - corr: Array type object. Correlation output vector.
    '''

    #check lenght and executes zero pad if needed
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    
    N = max(len(x1), len(x2))
    x1 = np.pad(x1, (0, N - len(x1)))
    x2 = np.pad(x2, (0, N - len(x2)))
    nfft = 2 * N

    freqs, x1_fft = get_fft(x1, fs, normalize=False, nfft=nfft, output="complex")
    freqs, x2_fft = get_fft(x2, fs, normalize=False, nfft=nfft, output="complex")
    cross_spect = x1_fft * np.conj(x2_fft)
      

    #weightings
    if isinstance(mode, str):
        if mode == "Classic":
            G = cross_spect
        elif mode == "Roth":
            psi = filters.roth(x2_fft)
            G = cross_spect * psi
        elif mode == "Scot":
            psi = filters.scot(x1_fft, x2_fft)
            G = cross_spect * psi
        elif mode == "PHAT":
            psi = filters.phat(cross_spect)
            G = cross_spect * psi
        else:
            raise ValueError('mode parameter must be either "Classic", "Roth", "Scot" or "PHAT".')
        corr = get_ifft(G, input = "complex", nfft = nfft)
        corr = np.round(np.real(np.fft.fftshift(corr)))
        corr = np.roll(corr, -1)
    else:
        raise ValueError('mode parameter must be a String object and either "Classic", "Roth", "Scot" or "PHAT".')
    return corr


def get_tau(mic_1, mic_2, fs=44100, mode="Classic"):
    """
    Gets the arrival time diference between 2 microphones
    Input:
        - mic_1: array type object. Microhpone 1 signal.
        - mic_2: array type object. Microhpone 2 signal.
        - fs: 
    Output:
        t: float type object. Arrival time diference
    """
    corr = cross_corr(mic_2, mic_1, mode=mode)
    n_corr = np.arange(-len(mic_2) +1, len(mic_1))
    tau = (n_corr[np.argmax(corr)]/fs)
    return tau

def get_taus_n_mic(mic_signals, fs=44100, mode="Classic"):
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
        tau = get_tau(reference, mic_signals[i], fs=fs, mode=mode)
        taus.append(tau)
    
    return taus

def get_direction(d, t, c=340, fs=44100):
    """
    Returns direction of arrival between 2 microphones
    Input:
        - d: float type object. Distance between microphones.
        - t: float type object. Time arrival difference between microphones.
        - c: Int type object. Sound propagation speed.
        - fs: Int type object. Sample Frequency.
    """
    angle = np.arccos(c*t/d)
    angle = np.rad2deg(angle)
    return angle

def get_direction_n_signals(d,taus, c=343, fs=44100):
    """
    Calcula los ángulos estimados de llegada del sonido para múltiples micrófonos.
    
    Input:
        - taus: list of float. Lista de retardos respecto al micrófono de referencia.
        - d: float. Distancia entre micrófonos.
        - c: float. Velocidad del sonido (por defecto 343 m/s).
        - fs: int. Frecuencia de muestreo (por defecto 44100 Hz).
    
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

def synth_impulse_response(fs, reverb_time, noise_florr_level, A=1.0):
    """
    Generates a synthetic impulse response
    Input:
        - fs: int type object. Sample rate
        - reverb_time: float type object. Reverb time.
        - noise_florr_level: int type object. Noise floor presion level.
        - A: float type object. Exponential amplitude. Optional, 1.0 by default.
    Output:
        - t: array type object. Time vector
        - impulse_response: array type object. Impulse response vector
    """

    #error handling
    if type(fs) != int:
        raise ValueError("fs must be an integer")
    if type(reverb_time) != float:
        raise ValueError("reverb_time must be a float")
    if type(noise_florr_level) != int:
        raise ValueError("noise_floor_level must be a int")
    if type(A) != float:
        raise ValueError("A must be a float")

    #cómo genero n? --> n, t, lo q sea, es arbitrario. Tiene que ser mayor al tiempo de reverberación.
    dur = reverb_time + 0.25
    signal_length = int(dur*fs)
    t = np.linspace(0, dur, signal_length, endpoint=True)

    #generate noise
    noise = np.random.normal(0, 1, signal_length)

    #envelop
    tao = reverb_time/6.90
    envolvente = np.exp(-t/tao)

    #impulse response generator
    impulse_response = A*envolvente*noise + (10**(noise_florr_level/20))*noise
    impulse_response = impulse_response/ np.max(np.abs(impulse_response))

    return t, impulse_response

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

def generate_time_vector(dur, fs):
    """
    Generates a time vector:
    Inputs:
        - dur: float type object. Vector time duration
        - fs: int type object. Sample frequency.
    Outputs:
        - t: array type object. Time vector
    """

    t = np.linspace(0, dur, int(dur*fs))
    return t

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

def get_ifft(in_rfft, in_phases=False, input="mag-phase", nfft = None):
    """
    Performs an inverse fast Fourier transform of a real signal
    Input:
        - in_rfft_mag: array type object. It must contain only the positive frequencies of the spectrum of the signal.
        - in_phases: array type object. It must contain only the positive frequencies of the spectrum of the signal. If false, it assumes the phases of all components are 0º.
        - input: str type object. Input format, can be "mag-phase" or "complex", "mag_phase" by default. If "complex", there must not be in_phases kwarg.
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
    
    temp_signal = scfft.irfft(in_rfft, n = nfft) # Cambio de scfft.irfft a np.fft.ifft si input="complex"
    return temp_signal


def rir(tau=0.15, rir_A=0.05, fs=44100, phi=-120, duration=0.1):
    """
    Generates a synthetic impulse response

    Parameters
    ----------
    fs : int, optional
        Sampling frequency. The default is 44100.
    tau : float, optional
        The tau time parameter that controls the duration of the exponential envelope. The default is 0.15.
    phi : float, optional
        The noise floor level in decibels. The default is -80.

    Returns
    -------
    rir_synth : ndarray
        Array that contains the synthesized impulse response.
    fs : int
        Sampling frequency in Hz.
    t : ndarray
        Contains a sequence of time values that corresponds to the impulse response.
        
    Raises
    ------
    ValueError
        If tau or phi have invalid values.
        Noise floor must be a number

    """
    if not isinstance(tau, (int, float)) or tau <= 0:
        raise ValueError("Tau debe ser un numero positivo.")
        
    if not isinstance(phi, (int, float)):
        raise ValueError("El piso de ruido (phi) debe ser un número.")
    
    if phi > 0:
        raise ValueError("Phi debe ser un número negativo.")
        
    A = 1
    fs = 44100
    #Duracion del ruido
    t = np.linspace(0, duration, int(fs * duration))       
    #Genero ruido
    noise_signal = np.random.normal(0,rir_A,t.size)
    #Genero un piso de ruido, con una atenuacion dada por phi
    noise_floor = noise_signal  * (10**((phi)/20))
    #Genero la envolvente exponencial
    exp = A*np.exp(-t/tau)
    #Modulo la señal con la exponencial y el ruido. Sumo el piso de ruido. Tengo en cuenta escalon unitario en t0=0
    rir_synth = exp * noise_signal + noise_floor
    #Normalizo la señal
    #rir_synth = rir_synth / np.max(np.abs(rir_synth))
    
    return rir_synth


def apply_reverb_synth(mic_signals, fs=44100, tau=0.15, rir_A = 0.05, p_noise = 0.1, phi=-120, duration=0.1):
    """
    Aplica reverberación a una lista de señales de micrófonos.

    Parámetros
    ----------
    mic_signals : list of ndarray
        Lista de señales por micrófono.
    fs : int
        Frecuencia de muestreo.
    tau : float
        Constante de decaimiento de la envolvente exponencial (más alto = más larga la cola de reverberación).
    rir_A : float
        Cantidad de ruido, amplitud, que se le agrega al IR.
    p_noise : float
        Amplitud del ruido rosa.    
    phi : float
        Nivel del piso de ruido en dB (debe ser negativo).
    duration : float
        Duración total de cada señal en segundos.

    Retorna
    -------
    mic_signals_rir : list of ndarray
        Lista de señales con reverberación agregada.
    """
    mic_signals_rir = []
    
    for sig in mic_signals:
        #sig = np.asarray(sig).flatten()     # Pasa a array y lo deja en 1 Dimension

        # Ruido
        beta = 1  # 1 = ruido rosa, 0 = blanco, 2 = browniano
        # Generar ruido rosa
        attenuation = p_noise
        pink_noise = cn.powerlaw_psd_gaussian(beta, int(fs*duration)) * attenuation
        
        rir_synth = rir(tau=tau, fs=fs, phi=phi, duration=duration, rir_A=rir_A)  # Calculo el RIR sintetico
        # Convolución
        sig_full = signal.fftconvolve(sig, rir_synth, mode="full")
        signal_rir = sig_full[:(int(len(sig_full)/2) + 1)]
        #Sumo el pulso
        signal_rir = sig + signal_rir + pink_noise
        #Normalizo
        signal_rir = signal_rir / np.max(np.abs(signal_rir))
        mic_signals_rir.append(signal_rir)
      
    return mic_signals_rir
