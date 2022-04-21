#%%
import wfdb
from scipy.signal import butter, sosfilt, lfilter, filtfilt
import shutil
import os
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib as mpl


def MWA_from_name(function_name):
    return MWA_cumulative

#Fast implementation of moving window average with numpy's cumsum function


def MWA_cumulative(input_array, window_size):

    ret = np.cumsum(input_array, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]

    for i in range(1, window_size):
        ret[i-1] = ret[i-1] / i
    ret[window_size - 1:] = ret[window_size - 1:] / window_size

    return ret


def panPeakDetect(detection, fs):

    min_distance = int(0.25*fs)

    signal_peaks = [0]
    noise_peaks = []

    SPKI = 0.0
    NPKI = 0.0

    threshold_I1 = 0.0
    threshold_I2 = 0.0

    RR_missed = 0
    index = 0
    indexes = []

    missed_peaks = []
    peaks = []

    for i in range(len(detection)):

        if i > 0 and i < len(detection)-1:
            if detection[i-1] < detection[i] and detection[i+1] < detection[i]:
                peak = i
                peaks.append(i)

                if detection[peak] > threshold_I1 and (peak-signal_peaks[-1]) > 0.3*fs:

                    signal_peaks.append(peak)
                    indexes.append(index)
                    SPKI = 0.125*detection[signal_peaks[-1]] + 0.875*SPKI
                    if RR_missed != 0:
                        if signal_peaks[-1]-signal_peaks[-2] > RR_missed:
                            missed_section_peaks = peaks[indexes[-2] +
                                                         1:indexes[-1]]
                            missed_section_peaks2 = []
                            for missed_peak in missed_section_peaks:
                                if missed_peak-signal_peaks[-2] > min_distance and signal_peaks[-1]-missed_peak > min_distance and detection[missed_peak] > threshold_I2:
                                    missed_section_peaks2.append(missed_peak)

                            if len(missed_section_peaks2) > 0:
                                missed_peak = missed_section_peaks2[np.argmax(
                                    detection[missed_section_peaks2])]
                                missed_peaks.append(missed_peak)
                                signal_peaks.append(signal_peaks[-1])
                                signal_peaks[-2] = missed_peak

                else:
                    noise_peaks.append(peak)
                    NPKI = 0.125*detection[noise_peaks[-1]] + 0.875*NPKI

                threshold_I1 = NPKI + 0.25*(SPKI-NPKI)
                threshold_I2 = 0.5*threshold_I1

                if len(signal_peaks) > 8:
                    RR = np.diff(signal_peaks[-9:])
                    RR_ave = int(np.mean(RR))
                    RR_missed = int(1.66*RR_ave)

                index = index+1

    signal_peaks.pop(0)

    return signal_peaks


plt.rcParams['figure.figsize'] = [15, 4] #ajustamos el tamaño de la gráfica

sig, fields = wfdb.rdsamp(
    'sample-data/mit-bih-arrhythmia-database-1.0.0/100', sampfrom=0, sampto=1500)
#rdsamp es un elemento de la biblioteca wfdb para leer la señal

unfiltered_signal = sig[:, 0] #leemos la señal de inicio a fin de la primera columna

plt.title("Señal sin filtrar de ECG")
plt.plot(unfiltered_signal, 'g') #cargamos el  vector a imprimir y el formato
plt.show()

#filtro paso banda
fs = 360  # sampling frequency of the MIT-BIH database (All insamples)
nyquist_freq = 0.5 * fs
f1 = 5/nyquist_freq  # Lowpass filtro paso baja
f2 = 15/nyquist_freq  # Highpass filtro paso alta

# cogemos las señales de ambos filtros, de la muestra inicial
# filtro paso alta
b, a = butter(1, f2*2, 'highpass')
filtered_ecg = filtfilt(b, a, unfiltered_signal)
plt.plot(filtered_ecg, 'g')
plt.title("ECG signal High-Pass")
plt.show()

#filtro paso baja
b, a = butter(1, f1*2, 'lowpass')
filtered_ecg = filtfilt(b, a, unfiltered_signal)
plt.plot(filtered_ecg, 'g')
plt.title("ECG signal Low-Pass")
plt.show()

# filtro banda que mezcla alta y baja

# The Butterworth filter is a type of signal processing filter designed to have as flat frequency response as possible(no ripples) in the pass-band
b, a = butter(1, [f1*2, f2*2], btype='bandpass')
filtered_ecg = lfilter(b, a, unfiltered_signal) #cogemos las señales de ambos filtros, de la muestra inicial
plt.plot(filtered_ecg, 'g')
plt.title("ECG signal Band-Pass")
plt.show()

# Derivación para eliminar para elimianr olas en componentes P y T del ECG
diff = np.diff(filtered_ecg)  # Diff is a numpy function for differentiation.
plt.plot(diff, 'g')
plt.title("ECG signal after differentiation")
plt.show()

# elevación al cuadrado
squared = diff*diff  # point to point squaring of the results from thederivation process.
plt.plot(squared, 'g')
plt.title("ECG Signal after squaring")
plt.show()

#detector
N = int(0.12*fs)
mwa = MWA_from_name("cumulative")(squared, N)
mwa[:int(0.2*fs)] = 0
mwa_peaks = panPeakDetect(mwa, fs)
plt.figure(figsize=(15, 3))
plt.plot(unfiltered_signal, 'g')
plt.plot(mwa_peaks, unfiltered_signal[mwa_peaks], "o")
plt.title("Raw (unfiltered) ECG signal with QRS marked")
plt.show()

# %%
