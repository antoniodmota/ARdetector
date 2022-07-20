#%%
import wfdb
import scipy.signal as sg
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from heartrate import heart_rate
import warnings
from tensorflow.python.keras.models import load_model
from sklearn.preprocessing import StandardScaler

invalid_beat = [
    "[", "!", "]", "x", "(", ")", "p", "t",
    "u", "`", "'", "^", "|", "~", "+", "s",
    "T", "*", "D", "=", '"', "@"
]

abnormal_beats = [
    "L", "R", "B", "A", "a", "J", "S", "V",
    "r", "F", "e", "j", "n", "E", "/", "f", "Q", "?"
]

warnings.filterwarnings(action='ignore', message='Mean of empty slice.')
warnings.filterwarnings(
    action='ignore', message='invalid value encountered in double_scalars')

def bandPass(signal):

    # Filtro paso banda
    fs = fields['fs']  # Extraemos frecuencia
    nyquist_freq = 0.5 * fs
    f1 = 5/nyquist_freq  # Lowpass filtro paso baja
    f2 = 15/nyquist_freq  # Highpass filtro paso alta
    # Filtro banda que mezcla alta y baja

    # Pasamos el filtro Butterworth para pasar el paso banda y dejar la señal lo más plana posible
    b, a = sg.butter(1, [f1*2, f2*2], btype='bandpass')
    # Cogemos las señales de ambos filtros, de la muestra inicial
    filtered_ecg = sg.lfilter(b, a, signal)
    
    return filtered_ecg

def lowPass(signal):

    # Filtro paso baja
    fs = 360  # Extraemos frecuencia
    nyquist_freq = 0.5 * fs
    f1 = 5/nyquist_freq  # Lowpass filtro paso baja
    # Filtro banda que mezcla alta y baja

    # Filtro paso baja
    b, a = sg.butter(1, f1*2, 'lowpass')
    filtered_ecg = sg.filtfilt(b, a, unfiltered_signal)

    return filtered_ecg

def highPass(signal):

    # Filtro paso alta
    fs = 360  # Extraemos frecuencia
    nyquist_freq = 0.5 * fs
    f2 = 15/nyquist_freq  # Highpass filtro paso alta

    b, a = sg.butter(1, f2*2, 'highpass')
    filtered_ecg = sg.filtfilt(b, a, unfiltered_signal)

    return filtered_ecg

def differentiation(signal):
    # Realizamos la derivada de la señal
    return np.diff(signal)

def squared(signal):
    # Realizamos el cuadrado de la señal
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

    pre_r = np.array(waves["ECG_R_Onsets"])
    post_r = np.array(waves["ECG_R_Offsets"])
    qrs_duration = post_r - pre_r

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

def PRsegment(waves):

    post_p = np.array(waves["ECG_P_Offsets"])
    pre_q = np.array(waves["ECG_R_Onsets"])
    prsg = pre_q - post_p

    return prsg

def PRinterval(waves):

    pre_p = np.array(waves["ECG_P_Onsets"])
    post_p = np.array(waves["ECG_P_Offsets"])
    p_duration = post_p - pre_p

    pr_interval = PRsegment(waves) + p_duration
    
    return pr_interval

def getSequence(signal, beat, window_sec, fs):
    window_size = window_sec * fs
    beat_start = beat - window_size
    beat_end = beat + window_size
    if beat_end < signal.shape[0]:
        sequence = signal[beat_start:beat_end, 0]
        return sequence.reshape(1, -1, 1)
    else:
        return np.array([])

def classifyBeat(symbol):
    if symbol in abnormal_beats:
        return 1
    elif symbol == "N" or symbol == ".":
        return 0

def buildDataset(all_sequences):
    sequences = []
    
    for i in all_sequences:
        sequences.append(i)

    return np.vstack(sequences)

# Ajustamos el tamaño de la gráfica
plt.rcParams['figure.figsize'] = [15, 4] 

print("\n Bienvenido a ARdetector! \n")

# Cargamos los pacientes existentes en la bbdd
pacient = np.loadtxt(
    "sample-data/mit-bih-arrhythmia-database-1.0.0/RECORDS", dtype=int)
pacient_list=list(pacient)
pacient_input=0

while pacient_input != -1 or pacient_input != -1:

    print("\nIndique el paciente a leer de la base de datos entre los siguientes o indique -1 para terminar: \n", pacient_list, "\n")
    pacient_input = int(input())

    if pacient_input in pacient_list:
        print("\nCargando paciente encontrado \n")

        # rdsamp es un elemento de la biblioteca wfdb para leer la señal
        sig, fields = wfdb.rdsamp(
            f'sample-data/mit-bih-arrhythmia-database-1.0.0/{pacient_input}')

        annotation = wfdb.rdann(
            f'sample-data/mit-bih-arrhythmia-database-1.0.0/{pacient_input}', 'atr')

        print("El total de tiempo muestreado es ",
              fields['sig_len']/fields['fs'], " segundos")
        print("El total de muestras es ", fields['sig_len'])

        
        # Leemos la señal de inicio a fin de la primera columna
        unfiltered_signal = sig[:, 0]
        final_unfiltered = unfiltered_signal

        # Operaciones de filtrado de señal
        filtered_ecg = bandPass(unfiltered_signal)
        diff = differentiation(filtered_ecg)
        sqrd = squared(diff)


        show = 0
        str_aux= "alguna"

        while show != -1 :

            print("\n¿Desea imprimir ", str_aux, " de las siguiente gráficas de filtrado de señal?, si no pulse -1\n 1)- Señal sin filtrar de ECG\n 2)- ECG filtro Paso-Banda\n 3)- Derivación de la señal\n 4)- Elevación al cuadrado\n")
            show = int(input())   

            if show == 1:
                # Cargamos el  vector a imprimir y el formato
                plt.title("Señal sin filtrar de ECG")
                plt.plot(final_unfiltered, 'g')
                plt.xlabel('Samples')
                plt.ylabel('MLIImV')
                plt.show()
                str_aux = "otra"

            if show == 2:
                # Filtro de paso banda
                plt.plot(filtered_ecg, 'g')
                plt.title("ECG filtro Paso-Banda")
                plt.xlabel('Samples')
                plt.ylabel('MLIImV')
                plt.show()
                str_aux = "otra"


            if show == 3:
                # Derivacion para eliminar para elimianr olas en componentes P y T del ECG
                plt.plot(diff, 'g')
                plt.title("ECG tras derivacion")
                plt.xlabel('Samples')
                plt.ylabel('MLIImV')
                plt.show()
                str_aux = "otra"
                
            if show == 4:

                # Elevacion al cuadrado
                plt.plot(sqrd, 'g')
                plt.title("ECG señal tras elevación al cuadrado")
                plt.show()
                str_aux = "otra"

            if show != -1 and show != 1 and show != 2 and show != 3 and show != 4:
                
                print("\nPor favor introduzca una opción correcta\n")


        print("\nCalulando características de la señal filtrada...\n")

        # Moving Window Integration (funcion de ecg-detectors)
        mwa = moving_window_integration(sqrd,  int(0.12 * fields['fs']))
        mwa[: int(0.2 * fields['fs'])] = 0


        # Procedemos a montar el detector
        # Buscamos los picos R con la clase heartrate
        hr = heart_rate(final_unfiltered, fields['fs'], mwa, filtered_ecg)
        result = hr.find_r_peaks()
        result = list(set(result))
        result.sort()
        result = np.array(result)
        result = result[result > 0]
        # Eliminamos los resultados donde la intensidad de la señal se menor que 0
        # puesto que se corresponde con el periodo de aprendizaje del algoritmo de picos
        for i in result:
            if final_unfiltered[i] <= 0:
                result = np.setdiff1d(result, i)

        
        

        # Se calcula la duración media del QRS
        # Pasamos a extraer el intervalo los puntos PQST utilizando la función ecg_delineate de neurokit2
        _, waves_peak = nk.ecg_delineate(sig[0:, 0],
                                {"ECG_R_Peaks": result},
                                360, method="dwt")

        # Se calculan los latidos
        heartRate = (60*fields['fs'])/np.average(np.diff(result[1:]))
        print("\nLatidos por minuto en media --> ", heartRate,' lpm')

        # Se calcula la duración media del intervalo RR
        RRdiff = RR(result) 
        RRmean = np.nanmean(RRdiff) / fields['fs']
        print("\nDuración media RR en        --> ", RRmean*1000,' ms')

        # Duracion media de QRS ---
        qrs_duration = QRS(waves_peak)
        np.delete(qrs_duration, qrs_duration.size -1)
       
        # Se calcula la media de QRS
        qrsMean = np.nanmean(qrs_duration) / fields['fs']
        print("\nDuración media QRS en ms    --> ", qrsMean*1000), ' ms'

        # Se calcula la duración media del segmento ST
        STsegdiff = STsegment(waves_peak)
        STsegmean = np.nanmean(STsegdiff) / fields['fs']
        print("\nDuración media segmento ST  --> ", STsegmean*1000, ' ms')

        # Se calcula la duración media del intervalo ST
        STintdiff = STinterval(waves_peak)
        STintmean = np.nanmean(STintdiff) / fields['fs']
        print("\nDuración media intervalo ST  --> ", STintmean*1000, ' ms')

        # Se calcula la duración media del segmento PR
        PRsegdiff = PRsegment(waves_peak)
        PRsegmean = np.nanmean(PRsegdiff) / fields['fs']
        print("\nDuración media segmento PR  --> ", PRsegmean*1000, ' ms')

        # Se calcula la duración media del intervalo PR
        PRintdiff = PRinterval(waves_peak)
        PRintmean = np.nanmean(PRintdiff) / fields['fs']
        print("\nDuración media intervalo PR  --> ", PRintmean*1000, ' ms')

        show = 0
        while show != -1:

            print("\n¿Desea imprimir ", str_aux,
                  " de las siguiente gráficas de filtrado de señal?, si no pulse -1\n \n1)- Localización de los picos R\n2)- Puntos caracteristicos TPQS\n")
            show = int(input())

            if show == 1:

                # Mostramos los picos R en la señal sin filtrar
                plt.figure(figsize=(20, 4), dpi=100)
                plt.xticks(np.arange(0, len(final_unfiltered)+1, 150))
                plt.plot(final_unfiltered)
                plt.scatter(result, final_unfiltered[result], color='red', s=50, marker='*')
                plt.xlabel('Samples')
                plt.ylabel('MLIImV')
                plt.title("Localización de los picos R")
                plt.show()

            if show == 2:

                # Pintamos los puntos caracteristicos TPQS
                plot = nk.events_plot([waves_peak['ECG_T_Peaks'][:],
                waves_peak['ECG_P_Peaks'][:],
                waves_peak['ECG_Q_Peaks'][:],
                waves_peak['ECG_S_Peaks'][:]], final_unfiltered[:])
                plt.show()

            if show != -1 and show != 1 and show != 2:

                print("\nPor favor introduzca una opción correcta\n")

        print("\nAplicamos la red neuronal CNN a la señal...\n")

        # Calsificador cnn
        record = wfdb.rdrecord(
            f'sample-data/mit-bih-arrhythmia-database-1.0.0/{pacient_input}')
        annotation = wfdb.rdann(
            f'sample-data/mit-bih-arrhythmia-database-1.0.0/{pacient_input}', 'atr')


        cnn = load_model('./modelo/modelo.h5')
        cnn.load_weights('./modelo/pesos.h5')

        scaler = StandardScaler()
        signal = scaler.fit_transform(record.p_signal)
        all_sequences = []
        # Resultados una vez quitados los que no se pueden extraer
        final_result = []

        # Llamamos a crear la secuecia de latidos con los picos R extraidos
        for i,i_sample in enumerate(result):
            sequence = getSequence(signal, i_sample, 3, 360)
            if sequence.size > 0:
                all_sequences.append(sequence)
                final_result.append(i_sample)

        # Construimos la estructura de datos para informar la red cnn        
        dataset = buildDataset(all_sequences)

        # Predecimos con el modelo previamente entrenado
        arreglo = cnn.predict(dataset)
        
        # Extraemos las posiciones dónde la probabilidad de arritmia es mayor de 0.5
        posAR=[]

        for i in arreglo:
            if i > 0.5:
                posAR.append(np.where(arreglo == i)[0])

        print('\nLos latidos en los que se pueden encontrar anomalias son: \n')

    
        # Picos r de latidos anormales
        resultPeak = []
        for i in posAR:
            print('\n Latido nº', i[0], " secuencia -> ", final_result[i[0]])
            resultPeak.append(final_result[i[0]])



        if not posAR:
            print("\nNo se observan latidos con anomalias\n")
    
        else:
            
            print("\n¿Desea imprimir la gráfica localizando los latidos irregulares? \n-1 - No\n 1 - Sí\n")
            show = int(input())

            if show == 1:

                # Mostramos los picos R de los latidos anormales en la señal sin filtrar
                plt.figure(figsize=(20, 4), dpi=100)
                plt.xticks(np.arange(0, len(final_unfiltered)+1, 150))
                plt.plot(final_unfiltered)
                plt.scatter(
                    resultPeak, final_unfiltered[resultPeak], color='red', s=50, marker='*')

                for i in resultPeak:
                    # Plotting a vertical line
                    plt.axvline(x=i, color='red')

                plt.xlabel('Samples')
                plt.ylabel('MLIImV')
                plt.title("Localización de los picos R")
                plt.show()

            # Pasamos a extraer el intervalo los puntos PQST utilizando la función ecg_delineate de neurokit2 para los resultados 
            # finales extraidos

            print('\n...Cargando datos...\n')
            _, waves_peak = nk.ecg_delineate(sig[0:, 0],
                                         {"ECG_R_Peaks": final_result},
                                         360, method="dwt")

            # Inicializamos la variable de entrada
            beat = 0

            while beat != -1:

                print("\n¿Desea imprimir alguno de los latidos irregulares junto con sus métricas? En caso afirmativo indique uno de la lista, en caso contrario indique -1\n") 

                for i in posAR:
                    print(i)

                print('\n')
                beat = int(input())

                # Mostramos las metricas y grafica
                if beat != -1 and beat in posAR:

                    print('\nLas metricas del latido indicado son:\n')

                    print('\nDuración QRS          --> ',
                          (waves_peak["ECG_R_Offsets"][beat]-waves_peak["ECG_R_Onsets"][beat])/360 * 1000, ' ms')

                    print('\nDuración segmento ST  --> ',
                          (waves_peak["ECG_T_Onsets"][beat]-waves_peak["ECG_R_Offsets"][beat])/360 * 1000, ' ms')

                    print('\nDuración intervalo ST --> ',
                          (waves_peak["ECG_T_Offsets"][beat]-waves_peak["ECG_R_Offsets"][beat])/360 * 1000, ' ms')

                    
                    if not np.isnan(waves_peak["ECG_P_Peaks"][beat]):
                        print('\nDuración segmento PR  --> ',
                              ( - waves_peak["ECG_P_Offsets"][beat] + waves_peak["ECG_R_Onsets"][beat])/360 * 1000, ' ms')

                        print('\nDuración intervalo PR --> ',
                          ( - waves_peak["ECG_P_Onsets"][beat]+waves_peak["ECG_R_Onsets"][beat])/360 * 1000, ' ms')
                    else:
                        print('\nNo hay presencia de onda P\n')
                  
                    secuence = sig[final_result[beat]-360:final_result[beat]+360,0]
                    # Mostramos señal ecg sin filtrar
                    plt.title("Señal del latido indicado")
                    plt.plot(secuence, 'g')
                    plt.xlabel('Samples')
                    plt.ylabel('MLIImV')
                    plt.show()



    elif pacient_input != -1 :
        print("\nEl paciente ", pacient_input, " no existe, elija uno correcto \n")



# %%
