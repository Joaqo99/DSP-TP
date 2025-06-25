import audio_functions as auf
import numpy as np
from scipy import signal

def roth(x1_spectrum):
    """
    Performs Roth weighting. Usefull to remove frequencies where Gn1n1 is large.
    Input:
        - x1: array type object. Reference mic spectrum signal.
        - fs: int type object. Sample frequency
    """
    gx1x1 = (np.abs(x1_spectrum))**2
    psi = 1/gx1x1
    return psi

def scot(x1_spectrum, x2_spectrum):
    """
    Performs Scot weighting. Usefull to remove frequencies where Gn2n2 or Gn1n1 are large.
    Input:
        - x1_spectrum: array type object. Reference mic signal spectrum.
        - x2_spectrum: array type object. Comparison mic signal spectrum.
    """
    gx1x1 = (np.abs(x1_spectrum))**2
    gx2x2 = (np.abs(x2_spectrum))**2
    psi = 1/np.sqrt(gx1x1*gx2x2)
    return psi

def phat(phi):
    """
    Performs PATH weighting.
    Input:
        - phi: array_type obecjt. Cross spectrum from signal 1 and 2.
        - fs: int type object. Sample frequency.
    """
    psi = 1/np.abs(phi)
    return psi

def eckart(X1_fft, X2_fft, x1, x2, snr_db = 15):
    """
    Calcula psi de Eckart usando PSD teórico de ruido a partir de SNR conocida.
    - X1_fft, X2_fft: arrays complejos, FFT completas de x1_padded y x2_padded (longitud N).
    - x1, x2: señales temporales originales (longitud L), usadas para potencia total.
    - snr_db: SNR en decibeles (misma en ambos canales).
    Retorna psi array de longitud N.
    """
    # Calculo potencias
    P_tot1 = np.mean(x1**2)
    P_tot2 = np.mean(x2**2)
    # SNR a lineal
    snr = 10**(snr_db/10)
    # Separo P_signal y P_noise
    #    P_n = P_total / (snr + 1)
    Gn1n1 = P_tot1 / (snr + 1)
    Gn2n2 = P_tot2 / (snr + 1)
    # PSD de señal útil: aproximamos con PSD total |X1_fft|^2
    Gs1s1 = np.abs(X1_fft)**2
    # Peso Eckart
    psi = Gs1s1 / (Gn1n1 * Gn2n2 + 1e-15)
    
    return psi
    
def ht(x1_spectrum, x2_spectrum, phi):
    """
    Performs HT / ML weighting.
    
    Parámetros:
        - x1_spectrum: FFT de la señal 1
        - x2_spectrum: FFT de la señal 2
        - phi: cross-spectrum: X1 * conj(X2)
    
    Retorna:
        - psi: ponderación espectral para aplicar a phi
    """

    gx1x1 = np.abs(x1_spectrum)**2
    gx2x2 = np.abs(x2_spectrum)**2
    epsilon = 1e-12
    gamma_12 = phi / (np.sqrt(gx1x1 * gx2x2) + epsilon)

    gamma_squared = np.clip(np.abs(gamma_12)**2, 0, 1 - epsilon)  # evita dividir por cero

    # Peso HT
    psi = (1/(np.abs(phi) + epsilon))*(gamma_squared/(1-gamma_squared))

    return psi

def anti_alias_filter(c, d, fs, order=1):
    """
    Anti spatial aliasing filter.
    Input:
        - c: sound speed.
        - d: microphone separation distance.
    Output:
        -
    """
    fc = (c/2*d)/(fs*0.5)
    sos_filter = signal.butter(order, fc, btype="lowpass", output="sos")
    return sos_filter