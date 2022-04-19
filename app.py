#%%
import wfdb
import posixpath
import shutil
import os
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams['figure.figsize'] = [15, 4] #ajustamos el tamaño de la gráfica

sig, fields = wfdb.rdsamp(
    'sample-data/mit-bih-arrhythmia-database-1.0.0/100', sampfrom=0, sampto=1300)
#rdsamp es un elemento de la biblioteca wfdb para leer la señal

unfiltered_signal = sig[:, 0] #leemos la señal de inicio a fin de la primera columna

plt.title("Señal sin filtrar de ECG")
plt.plot(unfiltered_signal, 'g') #cargamos el  vector a imprimir y el formato
plt.show()
# %%
