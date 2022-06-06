#%%
from heartrate import heart_rate# %%
from signal import siginterrupt
import wfdb
import scipy.signal as sg
import scipy.stats
import shutil
import os
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import neurokit2 as nk
from heartrate import heart_rate
from wfdb import processing



def bandPass(signal):

    #filtro paso banda
    fs = fields['fs']  # extraemos frecuencia
    nyquist_freq = 0.5 * fs
    f1 = 5/nyquist_freq  # Lowpass filtro paso baja
    f2 = 15/nyquist_freq  # Highpass filtro paso alta
    # filtro banda que mezcla alta y baja

    # Pasamos el filtro Butterworth para pasar el paso banda y dejar la señal lo más plana posible
    b, a = sg.butter(1, [f1*2, f2*2], btype='bandpass')
    # cogemos las señales de ambos filtros, de la muestra inicial
    filtered_ecg = sg.lfilter(b, a, signal)
    
    return filtered_ecg

def lowPass(signal):

    #filtro paso baja
    fs = 360  # extraemos frecuencia
    nyquist_freq = 0.5 * fs
    f1 = 5/nyquist_freq  # Lowpass filtro paso baja
    # filtro banda que mezcla alta y baja

    #filtro paso baja
    b, a = sg.butter(1, f1*2, 'lowpass')
    filtered_ecg = sg.filtfilt(b, a, unfiltered_signal)

    return filtered_ecg

def highPass(signal):

    #filtro paso alta
    fs = 360  # extraemos frecuencia
    nyquist_freq = 0.5 * fs
    f2 = 15/nyquist_freq  # Highpass filtro paso alta

    b, a = sg.butter(1, f2*2, 'highpass')
    filtered_ecg = sg.filtfilt(b, a, unfiltered_signal)

    return filtered_ecg

def differentiation(signal):
    # Realizamos la derivada de la señal
    return np.diff(signal)

def squared(signal):
    #realizamos el cuadrado de la señal
    return signal*signal

def moving_window_integration(signal, window_size, **kwargs):

    """Based on https://github.com/berndporr/py-ecg-detectors/

        Optimized for vectorized computation.

        """

    window_size = int(window_size)

    # Scipy's uniform_filter1d is a fast and accurate way of computing
    # moving averages. By default it computes the averages of `window_size`
    # elements centered around each element in the input array, including
    # `(window_size - 1) // 2` elements after the current element (when
    # `window_size` is even, the extra element is taken from before). To
    # return causal moving averages, i.e. each output element is the average
    # of window_size input elements ending at that position, we use the
    # `origin` argument to shift the filter computation accordingly.
    mwa = scipy.ndimage.uniform_filter1d(
        signal, window_size, origin=(window_size - 1) // 2)

    # Compute actual moving averages for the first `window_size - 1` elements,
    # which the uniform_filter1d function computes using padding. We want
    # those output elements to be averages of only the input elements until
    # that position.
    head_size = min(window_size - 1, len(signal))
    mwa[:head_size] = np.cumsum(signal[:head_size]) / \
        np.linspace(1, head_size, head_size)

    return mwa

def RR(r_peak):
    
    RR_duration = [np.nan]

    for beat in range(len(r_peak)-1):
        interval = (r_peak[beat+1] - r_peak[beat])  # Calculammos distancia entre R picos
        RR_duration.append(interval)

    return np.array(RR_duration)
    
def QRS(waves):
    post_p = np.array(waves["ECG_R_Onsets"])
    pre_t = np.array(waves["ECG_R_Offsets"])
    qrs_duration = pre_t - post_p
    return qrs_duration

def STsegment(waves):
    post_q = np.array(waves["ECG_R_Offsets"])
    pre_t = np.array(waves["ECG_T_Onsets"])
    st_segment = pre_t - post_q

    return st_segment

def STinterval(waves):
    post_q = np.array(waves["ECG_R_Offsets"])
    post_t = np.array(waves["ECG_T_Offsets"])
    st_interval = post_t - post_q

    return st_interval


#ajustamos el tamaño de la gráfica
plt.rcParams['figure.figsize'] = [15, 4] 

#rdsamp es un elemento de la biblioteca wfdb para leer la señal
sig, fields = wfdb.rdsamp(
    'sample-data/mit-bih-arrhythmia-database-1.0.0/100', sampfrom=180, sampto=4000)



#leemos la señal de inicio a fin de la primera columna
unfiltered_signal = sig[:, 0] 
final_unfiltered = unfiltered_signal

#cargamos el  vector a imprimir y el formato
plt.title("Señal sin filtrar de ECG")
plt.plot(unfiltered_signal, 'g') 
plt.xlabel('Samples')
plt.ylabel('MLIImV')
plt.show()



# cogemos las señales de ambos filtros, de la muestra inicial

# Filtro paso Alta
plt.plot(highPass(unfiltered_signal), 'g')
plt.title("ECG filtro Paso-Alta")
plt.xlabel('Samples')
plt.ylabel('MLIImV')
plt.show()

# Filtro paso baja
plt.plot(lowPass(unfiltered_signal), 'g')
plt.title("ECG filtro Paso-Baja")
plt.xlabel('Samples')
plt.ylabel('MLIImV')
plt.show()

# Filtro de paso banda 
filtered_ecg = bandPass(unfiltered_signal)
plt.plot(filtered_ecg, 'g')
plt.title("ECG filtro Paso-Banda")
plt.xlabel('Samples')
plt.ylabel('MLIImV')
plt.show()

# Derivación para eliminar para elimianr olas en componentes P y T del ECG
diff = differentiation(filtered_ecg)
plt.plot(diff, 'g')
plt.title("ECG tras derivacion")
plt.xlabel('Samples')
plt.ylabel('MLIImV')
plt.show()

# elevación al cuadrado
sqrd=squared(diff)
plt.plot(sqrd, 'g')
plt.title("ECG señal tras elevación al cuadrado")
plt.show()

# Moving Window Integration (funcion de ecg-detectors)
mwa = moving_window_integration(sqrd,  int(0.12 * fields['fs']))
mwa[: int(0.2 * fields['fs'])] = 0



# Procedemos a montar el detector
# Buscamos los picos R con la clase heartrate
hr = heart_rate(final_unfiltered, fields['fs'], mwa, filtered_ecg)
result = hr.find_r_peaks()
result = np.array(result)

# Eliminamos los resultados donde la intensidad de la señal se menor que 0
# puesto que se corresponde con el periodo de aprendizaje del algoritmo de picos
for i in result:
    if final_unfiltered[i] <= 0:
        result = np.setdiff1d(result, i)

# Mostramos los picos R en la señal sin filtrar
plt.figure(figsize=(20, 4), dpi=100)
plt.xticks(np.arange(0, len(final_unfiltered)+1, 150))
plt.plot(final_unfiltered)
plt.scatter(result, final_unfiltered[result], color='red', s=50, marker='*')
plt.xlabel('Samples')
plt.ylabel('MLIImV')
plt.title("R Peak Locations")
plt.show()

# Se calculan los latidos
heartRate = (60*fields['fs'])/np.average(np.diff(result[1:]))
print(heartRate, "Latidos por minuto")



# Pasamos a extraer el intervalo los puntos PQST utilizando la función ecg_delineate de neurokit2
_, waves_peak = nk.ecg_delineate(sig[0:, 0],
                                 {"ECG_R_Peaks": result},
                                 360, method="dwt")

# Pintamos los puntos caracteristicos TPQS
nk.events_plot([waves_peak['ECG_T_Peaks'][:],
                       waves_peak['ECG_P_Peaks'][:],
                       waves_peak['ECG_Q_Peaks'][:],
                waves_peak['ECG_S_Peaks'][:]], final_unfiltered[:4000])

# Duracion de los QRS
qrs_duration = QRS(waves_peak)

# Duracion de RR
RR_duration = RR(result)

# Duracion segmento e intervalo ST
st_interval= STinterval(waves_peak)
st_segment = STsegment(waves_peak)




# %%
