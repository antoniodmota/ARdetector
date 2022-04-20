#%%
import wfdb
from scipy.signal import butter, sosfilt, lfilter, filtfilt
import shutil
import os
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib as mpl

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
nyquist_freq=0.5 * fs
f1=5/nyquist_freq  # Lowpass filtro paso baja
f2=15/nyquist_freq  # Highpass filtro paso alta 
# The Butterworth filter is a type of signal processing filter designed to have as flat frequency response as possible(no ripples) in the pass-band
b, a = butter(1, [f1*2, f2*2], btype='bandpass')
filtered_ecg = lfilter(b, a, unfiltered_signal) #cogemos las señales de ambos filtros, de la muestra inicial
plt.plot(filtered_ecg, 'g')
plt.title("ECG signal Band-Pass")
plt.show()

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
# %%
