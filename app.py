#%%
from signal import siginterrupt
import wfdb
from scipy.signal import butter, sosfilt, lfilter, filtfilt
import shutil
import os
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from heartrate import heart_rate





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

def bandPass(signal):

    #filtro paso banda
    fs = 360  # sampling frequency of the MIT-BIH database (All insamples)
    nyquist_freq = 0.5 * fs
    f1 = 5/nyquist_freq  # Lowpass filtro paso baja
    f2 = 15/nyquist_freq  # Highpass filtro paso alta
    # filtro banda que mezcla alta y baja


    # The Butterworth filter is a type of signal processing filter designed to have as flat frequency response as possible(no ripples) in the pass-band
    b, a = butter(1, [f1*2, f2*2], btype='bandpass')
    # cogemos las señales de ambos filtros, de la muestra inicial
    filtered_ecg = lfilter(b, a, signal)
    

    return filtered_ecg

def lowPass(signal):

    #filtro paso baja
    fs = 360  # sampling frequency of the MIT-BIH database (All insamples)
    nyquist_freq = 0.5 * fs
    f1 = 5/nyquist_freq  # Lowpass filtro paso baja
    # filtro banda que mezcla alta y baja

    #filtro paso baja
    b, a = butter(1, f1*2, 'lowpass')
    filtered_ecg = filtfilt(b, a, unfiltered_signal)

    return filtered_ecg

def highPass(signal):

    #filtro paso alta
    fs = 360  # sampling frequency of the MIT-BIH database (All insamples)
    nyquist_freq = 0.5 * fs
    f2 = 15/nyquist_freq  # Highpass filtro paso alta

    b, a = butter(1, f2*2, 'highpass')
    filtered_ecg = filtfilt(b, a, unfiltered_signal)

    return filtered_ecg

def differentiation(signal):

    return np.diff(signal)

def squared(signal):
    return signal*signal

def moving_window_integration(signal, anotation):
    '''
    Moving Window Integrator
    :param signal: input signal
    :return: prcoessed signal

    Methodology/Explaination:
    The moving window integration process is done to obtain
    information about both the slope and width of the QRS complex.
    A window size of 0.15*(sample frequency) is used for more
    accurate results.

    The moving window integration has the recursive equation:
      y(nT) = [y(nT - (N-1)T) + x(nT - (N-2)T) + ... + x(nT)]/N

      where N is the number of samples in the width of integration
      window.
    '''

    # Initialize result and window size for integration
    result = signal.copy()
    win_size = round(0.150 * annotation.fs)
    sum = 0

    # Calculate the sum for the first N terms
    for j in range(win_size):
      sum += signal[j]/win_size
      result[j] = sum

    # Apply the moving window integration using the equation given
    for index in range(win_size, len(signal)):
      sum += signal[index]/win_size
      sum -= signal[index-win_size]/win_size
      result[index] = sum

    return result

def find_r_peaks(self):
    '''
    R Peak Detection
    '''

    # Find approximate peak locations
    self.approx_peak()

    # Iterate over possible peak locations
    for ind in range(len(self.peaks)):

        # Initialize the search window for peak detection
        peak_val = self.peaks[ind]
        win_300ms = np.arange(max(0, self.peaks[ind] - self.win_150ms), min(
            self.peaks[ind] + self.win_150ms, len(self.b_pass)-1), 1)
        max_val = max(self.b_pass[win_300ms], default=0)

        # Find the x location of the max peak value
        if (max_val != 0):
          x_coord = np.asarray(self.b_pass == max_val).nonzero()
          self.probable_peaks.append(x_coord[0][0])

        if (ind < len(self.probable_peaks) and ind != 0):
            # Adjust RR interval and limits
            self.adjust_rr_interval(ind)

            # Adjust thresholds in case of irregular beats
            if (self.RR_Average1 < self.RR_Low_Limit or self.RR_Average1 > self.RR_Missed_Limit):
                self.Threshold_I1 /= 2
                self.Threshold_F1 /= 2

            RRn = self.RR1[-1]

            # Searchback
            self.searchback(peak_val, RRn, round(RRn*self.samp_freq))

            # T Wave Identification
            self.find_t_wave(peak_val, RRn, ind, ind-1)

        else:
          # Adjust threholds
          self.adjust_thresholds(peak_val, ind)

        # Update threholds for next iteration
        self.update_thresholds()

    # Searchback in ECG signal
    self.ecg_searchback()

    return self.result

#ajustamos el tamaño de la gráfica
plt.rcParams['figure.figsize'] = [15, 4] 

#rdsamp es un elemento de la biblioteca wfdb para leer la señal
sig, fields = wfdb.rdsamp(
    'sample-data/mit-bih-arrhythmia-database-1.0.0/100', sampfrom=180, sampto=4000)
annotation = wfdb.rdann('sample-data/mit-bih-arrhythmia-database-1.0.0/100', 'atr', sampfrom=180,
                        sampto=4000, shift_samps=True)

record = wfdb.rdrecord(
    'sample-data/mit-bih-arrhythmia-database-1.0.0/100', sampfrom=180, sampto=4000,)


#leemos la señal de inicio a fin de la primera columna
unfiltered_signal = sig[:, 0] 
final_unfiltered = unfiltered_signal
#cargamos el  vector a imprimir y el formato
plt.title("Señal sin filtrar de ECG")
plt.plot(unfiltered_signal, 'g') 
plt.show()



# cogemos las señales de ambos filtros, de la muestra inicial

# Filtro paso Alta
plt.plot(highPass(unfiltered_signal), 'g')
plt.title("ECG signal High-Pass")
plt.show()

# Filtro paso baja
plt.plot(lowPass(unfiltered_signal), 'g')
plt.title("ECG signal Low-Pass")
plt.show()

# Filtro de paso banda 
filtered_ecg = bandPass(unfiltered_signal)
plt.plot(filtered_ecg, 'g')
plt.title("ECG signal Band-Pass")
plt.show()

# Derivación para eliminar para elimianr olas en componentes P y T del ECG
  # Diff is a numpy function for differentiation.
diff = differentiation(filtered_ecg)
plt.plot(diff, 'g')
plt.title("ECG signal after differentiation")
plt.show()

# elevación al cuadrado
sqrd=squared(diff)
plt.plot(sqrd, 'g')
plt.title("ECG Signal after squaring")
plt.show()

# Moving Window Integration Function

mwin = moving_window_integration(sqrd,annotation)
mwin[:int(0.2*360)] = 0
mwa_peaks = panPeakDetect(mwin, 360)
plt.figure(figsize=(15, 3))
plt.plot(unfiltered_signal, 'g')
plt.plot(mwa_peaks, final_unfiltered[mwa_peaks], "o")
plt.title("Raw (unfiltered) ECG signal with QRS marked")
plt.show()

'''
#detector
fs=360 #frecuencia de muestreo
N = int(0.12*fs)
## otro algoritomo de moving windows
mwa = MWA_cumulative(sqrd, N)
mwa[:int(0.2*fs)] = 0
mwa_peaks = panPeakDetect(mwa, fs)
plt.figure(figsize=(15, 3))
plt.plot(unfiltered_signal, 'g')
plt.plot(mwa_peaks, final_unfiltered[mwa_peaks], "o")
plt.title("Raw (unfiltered) ECG signal with QRS marked")
plt.show()
'''
#procedemos a montar el detector
ecg = pd.DataFrame(np.array([list(range(len(sig))), sig[
                   :, 0]]).T, columns=['TimeStamp', 'ecg'])

# Convert ecg signal to numpy array
signal = ecg.iloc[:, 1].to_numpy()

# Find the R peak locations
hr = heart_rate(signal, annotation.fs, mwin, filtered_ecg)
result = hr.find_r_peaks()
result = np.array(result)

# Clip the x locations less than 0 (Learning Phase)
result = result[result > 0]

# Calculate the heart rate
heartRate = (60*annotation.fs)/np.average(np.diff(result[1:]))
print("Heart Rate", heartRate, "BPM")

# Plotting the R peak locations in ECG signal
plt.figure(figsize=(20, 4), dpi=100)
plt.xticks(np.arange(0, len(signal)+1, 150))
plt.plot(signal, color='blue')
plt.scatter(result, signal[result], color='red', s=50, marker='*')
plt.xlabel('Samples')
plt.ylabel('MLIImV')
plt.title("R Peak Locations")

# %%
