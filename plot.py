from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import signal
import numpy as np
import audio_functions as auf

nominal_oct_central_freqs = [31.5, 63, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 160000]


def plot_signal(*vectors, xticks=None, yticks=None, title=None, file_name=False, grid=False, log=False, figsize=False, show=True, y_label="Amplitud", xlimits = False, ylimits = False, legend=False):
    """
    Plots multiple time signals over the same plot.
    Input:
        - vectors: Optional amount of values. For each vector: Dict type object. Must contain:
            - time vector: array type object. Time vector.
            - signal: array or type object. Amplitudes vector.
            - label: str type object. 
            - color: string type object.

        - xticks: Optional. Int type object.
        - yticks: array type object. Optional
        - title: string type object. Optional
        - file_name: string type object. Optional. If true, saves the figure in graficos folder.
        - grid: boolean type object. Optional.
        - log: boolean type object. Optional.
        - figsize: tuple of ints type object. Optional. In order to use with Multiplot function, it must be false.
        - show: Bool type object. If true, shows the plot. In order to use with Multiplot function, it must be false.
        - xlimits: tuple type object.
        - ylimits: tuple type object.
        - legend: bool type object. False by default.
    Output:
        - Signal plot
        - If file_name is true, saves the figure and prints a message.
    """
    if figsize:
        plt.figure(figsize=figsize) 

    if type(xticks) != int and type(xticks) != type(None):            
            raise Exception("xtick value must be an int")
    
    
    if type(xticks) == int:
        if xticks == 1:
            plt.xticks(np.arange(0, xticks + 0.1, 0.1))
        else:
            plt.xticks(np.arange(0, xticks+1, 1))

    for vector in vectors:

        #check keys

        #time vector
        if not ("time vector" in vector.keys()):
            raise Exception("time vector key missing")
        else:
            #turn to numpy
            n = vector["time vector"]
            if type(n) != np.ndarray:
                raise ValueError("Time vector must be an array or a Tensor")

        #signal vector
        if not ("signal" in vector.keys()):
            raise Exception("signal key missing")
        else:
            #turn to numpy
            signal = vector["signal"]
            if type(signal) != np.ndarray:
                raise ValueError("Audio signal must be an array or a Tensor")

        label = vector["label"] if "label" in vector.keys() else None
        color = vector["color"] if "color" in vector.keys() else None

        #plot signal
        plt.plot(n, signal, label=label, color=color)
        plt.xlabel("Tiempo [s]", fontsize=13)

    if type(yticks) == np.ndarray:
        if type(yticks) != np.ndarray:            
            raise Exception("ytick value must be an array")
        
        if not(ylimits):            
            plt.ylim(np.min(yticks), np.max(yticks))

        plt.yticks(yticks)

    plt.grid(grid)
    
    if xlimits:
        if type(xlimits) != tuple:
            raise ValueError("Xlimits must be tuple type")
        plt.xlim(xlimits)

    if ylimits:
        if type(ylimits) != tuple:
            raise ValueError("Xlimits must be tuple type")
        plt.ylim(ylimits)

    if log:
        plt.yscale("log")

    plt.ylabel(f"{y_label}", fontsize=13)

    if title:
        plt.title(title, fontsize=15)

    #save file
    if file_name:
        plt.savefig(f"../graficos/{file_name}.png")
        #print(f"File saved in graficos/{file_name}.png")
    
    if legend:
        plt.legend()

    if show: 
        plt.show()
    else:
        plt.ioff()

def plot_ftf(filters, fs, f_lim=False, figsize=False, show=True, title=False, xticks=nominal_oct_central_freqs):
    """
    Plots a filter transfer function
    Input:
        - filters: list of filters. Sos format required.
        - fs: int type object. sample rate
        - f_lim: list type object. Frequency visualization limits. False 
        - figsize: tuple of ints type object. Optional. In order to use with Multiplot function, it must be false.
        - show: Bool type object. If true, shows the plot. In order to use with Multiplot function, it must be false.
        - title: string type object. False by default.
        - xticks: structured type object. Ticks of frequencies.
    """
    if figsize:
        plt.figure(figsize=figsize)
    for sos in filters:
        wn, H = signal.sosfreqz(sos, worN=16384)
        f= (wn*(0.5*fs))/np.pi

        eps = np.finfo(float).eps

        H_mag = 20 * np.log10(abs(H) + eps)


        # #La magnitud de H se grafica usualmente en forma logarítmica
        plt.semilogx(f, H_mag)
    plt.xlabel('Frecuencia (Hz)')

    if f_lim:
        f_min, f_max = f_lim
        plt.xlim(f_min, f_max)
        xticks =list(filter(lambda f: f >= f_min and f <= f_max, xticks))
        plt.xticks(xticks, xticks)

    plt.ylim(-6,1)
    if title:
        plt.title(title)

    plt.ylabel('Magnitud [dB]')
    plt.grid()
    if show: 
        plt.show()
    else:
        plt.ioff()

def multiplot(*plots, figsize=(8, 5), ncols=2):
    """
    Receive single plots as lambda functions and subplots them all in a grid with the specified number of columns.
    Inputs:
        - plots: lambda function type objects. Each plot must have Show and Figsize arguments set to False.
        - figsize: tuple specifying figure size.
        - ncols: int, number of columns in the subplot grid.
    """
    import matplotlib.pyplot as plt
    num_plots = len(plots)
    rows = (num_plots + ncols - 1) // ncols  # Ceiling division
    plt.figure(figsize=figsize)
    for i, figure in enumerate(plots):
        plt.subplot(rows, ncols, i + 1)
        figure()
    plt.tight_layout()
    plt.show()

def plot_fft_mag(*in_signals, fs=44100, N=1, title=False, legend=False, show=True, xlimits = False, normalize=True, ylimits=False, figsize=False, xticks=False, grid=False):

    """
    Plots the magnitude of the fast fourier transform of an arbitrary ammount of audio signals.
    Inputs:
        - in_signals : Optional amount of values. For each signal: Dict type object. Must contain:
            - audio signal: array type object.
            - label: string type object.
            - color: string type object.
    """

    for in_signal in in_signals:
        #calcular fft

        if not ("audio signal" in in_signal.keys()):
            raise Exception("Audio signal key missing")
        else:
            audio_signal = in_signal["audio signal"]
            if type(audio_signal) != np.ndarray:
                raise ValueError("Audio signal must be an array or a Tensor")

        label = in_signal["label"] if "label" in in_signal.keys() else None
        color = in_signal["color"] if "color" in in_signal.keys() else None

        in_freqs, fft_mag_norm, _ = auf.get_fft(audio_signal, fs, normalize=normalize)
        eps = np.finfo(float).eps
        fft_mag_db = 20*np.log10(fft_mag_norm + eps)

            # Apply the moving average filter
        if N > 1:
            ir = np.ones(N) * 1 / N  # Moving average impulse response
            fft_mag_db = signal.fftconvolve(fft_mag_db, ir, mode='same')

    # Logarithmic scale for the x-axis
        plt.semilogx(in_freqs, fft_mag_db, label=label, color=color)


    # grafico fft
    if xticks:
        plt.xticks([t for t in xticks], [f'{t}' for t in xticks])
    else:
        ticks = [31, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        plt.xticks([t for t in ticks], [f'{t}' for t in ticks])
        plt.xlim(20, 22000)
    plt.ylim(-80, np.max(fft_mag_db) + 10)
    plt.xlabel("Frecuencia [Hz]", fontsize=13)
    plt.ylabel("Amplitud [dB]", fontsize=13)

    if xlimits:
        if type(xlimits) != tuple:
            raise ValueError("Xlimits must be tuple type")
        plt.xlim(xlimits)

    if ylimits:
        if type(ylimits) != tuple:
            raise ValueError("ylimits must be tuple type")
        plt.ylim(ylimits)

    if figsize:
        plt.figure(figsize=figsize)

    plt.grid(grid)

    if title:
        plt.title(title)

    if legend:
        plt.legend()

    if show: 
        plt.show()

def plot_fft_phase(*in_signals, fs=44100, N=1, title=False, legend=False, show=True, xlimits = False, ylimits=False, grid=False, figsize=False, xticks=False):

    """
    Plots the phase of the fast fourier transform of an arbitrary ammount of audio signals.
    Inputs:
        - in_signals : Optional amount of values. For each signal: Dict type object. Must contain:
            - audio signal: array type object.
            - label: string type object.
            - color: string type object.
    """

    for in_signal in in_signals:

        if not ("audio signal" in in_signal.keys()):
            raise Exception("Audio signal key missing")
        else:
            audio_signal = in_signal["audio signal"]
            if type(audio_signal) != np.ndarray:
                raise ValueError("Audio signal must be an array or a Tensor")

        label = in_signal["label"] if "label" in in_signal.keys() else None
        color = in_signal["color"] if "color" in in_signal.keys() else None

        in_freqs, _, fft_phase = auf.get_fft(audio_signal, fs)

    # Logarithmic scale for the x-axis
        plt.semilogx(in_freqs, fft_phase, label=label, color=color)


    # grafico fft
    if xticks:
        plt.xticks([t for t in xticks], [f'{t}' for t in xticks])
    else:
        ticks = [31, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        plt.xticks([t for t in ticks], [f'{t}' for t in ticks])
        plt.xlim(20, 22000)
    #plt.ylim(-80, np.max(fft_mag_db) + 10)
    plt.xlabel("Frecuencia [Hz]", fontsize=13)
    plt.ylabel("Amplitud [dB]", fontsize=13)

    if xlimits:
        if type(xlimits) != tuple:
            raise ValueError("Xlimits must be tuple type")
        plt.xlim(xlimits)

    if ylimits:
        if type(ylimits) != tuple:
            raise ValueError("ylimits must be tuple type")
        plt.ylim(ylimits)

    if figsize:
        plt.figure(figsize=figsize)

    if title:
        plt.title(title)

    if legend:
        plt.legend()

    plt.grid(grid)

    if show: 
        plt.show()

def unit_plot(*vectors, xticks=None, yticks=None, title=None, file_name=False, grid=False, log=False, figsize=False, show=True, y_label="", xlimits = False, ylimits = False, legend=False):
    """
    Plots a unities vector.
    Input:
        - vectors: Optional amount of values. For each vector: Dict type object. Must contain:
            - array: array type object. Amplitudes vector.
            - label: str type object. 
            - color: string type object.

        - xticks: Optional. Int type object.
        - yticks: array type object. Optional
        - title: string type object. Optional
        - file_name: string type object. Optional. If true, saves the figure in graficos folder.
        - grid: boolean type object. Optional.
        - log: boolean type object. Optional.
        - figsize: tuple of ints type object. Optional. In order to use with Multiplot function, it must be false.
        - show: Bool type object. If true, shows the plot. In order to use with Multiplot function, it must be false.
        - xlimits: tuple type object.
        - ylimits: tuple type object.
        - legend: bool type object. False by default.
    Output:
        - Signal plot
        - If file_name is true, saves the figure and prints a message.
    """
    if figsize:
        plt.figure(figsize=figsize) 

    if type(xticks) != int and type(xticks) != type(None):            
            raise Exception("xtick value must be an int")
    
    if type(xticks) == int:
        if xticks == 1:
            plt.xticks(np.arange(0, xticks + 0.1, 0.1))
        else:
            plt.xticks(np.arange(0, xticks+1, 1))

    for vector in vectors:

        #check keys

        #signal vector
        if not ("array" in vector.keys()):
            raise Exception("array key missing")
        else:
            #turn to numpy
            arr = vector["array"]
            if type(arr) != np.ndarray:
                raise ValueError("Array must be an ndarray or a Tensor")

        label = vector["label"] if "label" in vector.keys() else None
        color = vector["color"] if "color" in vector.keys() else None

        n = np.arange(len(arr))

        #plot signal
        plt.plot(n, arr, label=label, color=color)
        plt.xlabel("Unit", fontsize=13)

    if type(yticks) == np.ndarray:
        if type(yticks) != np.ndarray:            
            raise Exception("ytick value must be an array")
        
        if not(ylimits):            
            plt.ylim(np.min(yticks), np.max(yticks))

        plt.yticks(yticks)

    plt.grid(grid)
    
    if xlimits:
        if type(xlimits) != tuple:
            raise ValueError("Xlimits must be tuple type")
        plt.xlim(xlimits)

    if ylimits:
        if type(ylimits) != tuple:
            raise ValueError("Xlimits must be tuple type")
        plt.ylim(ylimits)

    if log:
        plt.yscale("log")

    plt.ylabel(f"{y_label}", fontsize=13)

    if title:
        plt.title(title, fontsize=15)

    #save file
    if file_name:
        plt.savefig(f"../graficos/{file_name}.png")
        #print(f"File saved in graficos/{file_name}.png")
    
    if legend:
        plt.legend()

    if show: 
        plt.show()
    else:
        plt.ioff()