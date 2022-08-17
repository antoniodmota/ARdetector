#%%
import wfdb
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Conv1D, MaxPooling1D

from tensorflow.python.keras.optimizer_v2 import adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix



invalid_beat = [
    "[", "!", "]", "x", "(", ")", "p", "t",
    "u", "`", "'", "^", "|", "~", "+", "s",
    "T", "*", "D", "=", '"', "@"
]

abnormal_beats = [
    "L", "R", "B", "A", "a", "J", "S", "V",
    "r", "F", "e", "j", "n", "E", "/", "f", "Q", "?"
]


def classifyBeat(beat):
    if beat in abnormal_beats:
        return 1
    elif beat == "N" or beat == ".":
        return 0


def getSequence(signal, beat_loc, sec, fs):
   
    # Calculamos el tamaño del intervalo de señal para el latido, sumando y restando 3s
    beat_start = beat_loc - (sec * fs)
    beat_end = beat_loc + (sec * fs)

    # Delvolvemos el subarray de muestras
    if beat_end < signal.shape[0]:
        sequence = signal[beat_start:beat_end, 0]
        return sequence.reshape(1, -1, 1)
    else:
        return np.array([])


def buildDataset(df, all_sequences, all_labels):
    
    sequences = []
    labels = []
    # Se extraen los pacientes que se indican en df del total de medidas
    for i, row in df.iterrows():
        sequences.extend(all_sequences[int(row["inicio"]):int(row["fin"])])
        labels.extend(all_labels[int(row["inicio"]):int(row["fin"])])

    return np.vstack(sequences), np.vstack(labels)


# Cargamos todos los pacientes que hay en la base de datos
pacient = np.loadtxt(
    "sample-data/mit-bih-arrhythmia-database-1.0.0/RECORDS", dtype=int)


all_sequences = []
all_beats = []
sec = 3
subject_map = []

for subject in pacient:
    record = wfdb.rdrecord(
        f'sample-data/mit-bih-arrhythmia-database-1.0.0/{subject}')
    annotation = wfdb.rdann(
        f'sample-data/mit-bih-arrhythmia-database-1.0.0/{subject}', 'atr')

    # Cargamos todos los muestreos y tipos de latidos
    atr_symbol = annotation.symbol
    atr_sample = annotation.sample
    fs = record.fs
    scaler = StandardScaler()
    signal = scaler.fit_transform(record.p_signal)
    subject_type_beats = []

    # Clasificamos los latidos y extraemos los sub segmentos de la muestra
    for i, i_smp in enumerate(atr_sample):
        type_beat = classifyBeat(atr_symbol[i])
        sequence = getSequence(signal, i_smp, sec, fs)

        if type_beat is not None and sequence.size > 0:
            all_sequences.append(sequence)
            subject_type_beats.append(type_beat)

    # Extraemos el porcentaje de latidos irreglares 
    normal_percentage = sum(subject_type_beats) / len(subject_type_beats)

    # Generamos un mapa para cada paciente 
    subject_map.append({
        "paciente": subject,
        "porcentaje": normal_percentage,
        "num_sec": len(subject_type_beats),
        "inicio": len(all_beats),
        "fin": len(all_beats)+len(subject_type_beats)
    })
    all_beats.extend(subject_type_beats)

subject_map = pd.DataFrame(subject_map)

# Estandarizamos los resultados del porcentaje de latidos iregulares para extraer posteriormente
# Conjuntos de datos homogeneos con los de validacion
bins = [0, 0.2, 0.6, 1.0]
subject_map["bin"] = pd.cut(
    subject_map['porcentaje'], bins=bins, labels=False, include_lowest=True)

# Extraemos conjunto de entrenamiento y validacion
train, validation = train_test_split(
    subject_map, test_size=0.25, stratify=subject_map["bin"], random_state=42)

X_train, y_train = buildDataset(train, all_sequences, all_beats)
X_val, y_val = buildDataset(validation, all_sequences, all_beats)
X_train.shape, y_train.shape


##CNN model

cnn_model = Sequential([
    Conv1D(
        filters=10,
        kernel_size=4,
        strides=1,
        input_shape=(X_train.shape[1], 1),
        padding="same",
        activation="relu"
    ),
    Conv1D(20, 4, activation="relu"),
    MaxPooling1D(3),
    Conv1D(10, 4, activation="relu"),
    MaxPooling1D(2),
    Flatten(),
    Dropout(0.5),
    Dense(
        1,
        activation="sigmoid",
        name="output",
    )
])

optimizer = adam.Adam(lr=0.001)

# Compilamos el modelo
cnn_model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
cnn_model.summary()

# Entrenamos modelo
trained_cnn = cnn_model.fit(
    X_train,
    y_train,
    batch_size=128,
    epochs=15,
    validation_data=(X_val, y_val)
)

# Extraemos metricas de la red
scores=cnn_model.evaluate(X_val, y_val)
print("\n%s: %.2f%%" % (cnn_model.metrics_names[1], scores[1]*100))
arreglo = cnn_model.predict(X_val)



# Mostramos grafica de precision
plt.plot(trained_cnn.history['accuracy'])
plt.plot(trained_cnn.history['val_accuracy'])
plt.title('CNN model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Mostramos grafica de perdida
plt.plot(trained_cnn.history['loss'])
plt.plot(trained_cnn.history['val_loss'])
plt.title('CNN model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Guardamos el modelo en un archivo

dir= './modelo/'

if not os.path.exists(dir):
    os.mkdir(dir)

cnn_model.save('./modelo/modelo.h5')
cnn_model.save_weights('./modelo/pesos.h5')

# %%
