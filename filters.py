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

def eckart(x1_spectrum, x2_spectrum, phi):
    """
    Performs Eckart weighting.
    Input:
        - x1_spectrum: array type object. Reference mic signal spectrum.
        - x2_spectrum: array type object. Comparison mic signal spectrum.
    """
    
    #gx1x2 = phi
    gx1x1 = (np.abs(x1_spectrum))**2
    gx2x2 = (np.abs(x2_spectrum))**2
    psi = np.abs(phi) * ((gx1x1 - np.abs(phi))*(gx2x2 - np.abs(phi)))
    return psi

    