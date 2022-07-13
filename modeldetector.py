#%%
from genericpath import exists
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

from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation, Conv1D
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizer_v2 import adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



invalid_beat = [
    "[", "!", "]", "x", "(", ")", "p", "t",
    "u", "`", "'", "^", "|", "~", "+", "s",
    "T", "*", "D", "=", '"', "@"
]

abnormal_beats = [
    "L", "R", "B", "A", "a", "J", "S", "V",
    "r", "F", "e", "j", "n", "E", "/", "f", "Q", "?"
]


def classify_beat(symbol):
    if symbol in abnormal_beats:
        return 1
    elif symbol == "N" or symbol == ".":
        return 0


def get_sequence(signal, beat_loc, window_sec, fs):
    window_one_side = window_sec * fs
    beat_start = beat_loc - window_one_side
    beat_end = beat_loc + window_one_side
    if beat_end < signal.shape[0]:
        sequence = signal[beat_start:beat_end, 0]
        return sequence.reshape(1, -1, 1)
    else:
        return np.array([])


def build_dataset(df, all_sequences, all_labels):
    sequences = []
    labels = []
    for i, row in df.iterrows():
        start = int(row["start"])
        end = int(row["end"])
        sequences.extend(all_sequences[start:end])
        labels.extend(all_labels[start:end])

    return np.vstack(sequences), np.vstack(labels)



pacient = np.loadtxt(
    "sample-data/mit-bih-arrhythmia-database-1.0.0/RECORDS", dtype=int)


all_sequences = []
all_labels = []
window_sec = 3
subject_map = []
for subject in pacient:
    record = wfdb.rdrecord(
        f'sample-data/mit-bih-arrhythmia-database-1.0.0/{subject}')
    annotation = wfdb.rdann(
        f'sample-data/mit-bih-arrhythmia-database-1.0.0/{subject}', 'atr')
    atr_symbol = annotation.symbol
    atr_sample = annotation.sample
    fs = record.fs
    scaler = StandardScaler()
    signal = scaler.fit_transform(record.p_signal)
    subject_labels = []
    for i, i_sample in enumerate(atr_sample):
        label = classify_beat(atr_symbol[i])
        sequence = get_sequence(signal, i_sample, window_sec, fs)
        if label is not None and sequence.size > 0:
            all_sequences.append(sequence)
            subject_labels.append(label)

    normal_percentage = sum(subject_labels) / len(subject_labels)
    subject_map.append({
        "subject": subject,
        "percentage": normal_percentage,
        "num_seq": len(subject_labels),
        "start": len(all_labels),
        "end": len(all_labels)+len(subject_labels)
    })
    all_labels.extend(subject_labels)

subject_map = pd.DataFrame(subject_map)


bins = [0, 0.2, 0.6, 1.0]
subject_map["bin"] = pd.cut(
    subject_map['percentage'], bins=bins, labels=False, include_lowest=True)

train, validation = train_test_split(
    subject_map, test_size=0.25, stratify=subject_map["bin"], random_state=42)

X_train, y_train = build_dataset(train, all_sequences, all_labels)
X_val, y_val = build_dataset(validation, all_sequences, all_labels)
X_train.shape, y_train.shape


##CNN model

sequence_size = X_train.shape[1]
n_features = 1

cnn_model = Sequential([
    Conv1D(
        filters=8,
        kernel_size=4,
        strides=1,
        input_shape=(sequence_size, n_features),
        padding="same",
        activation="relu"
    ),
    Flatten(),
    Dropout(0.5),
    Dense(
        1,
        activation="sigmoid",
        name="output",
    )
])

optimizer = adam.Adam(lr=0.001)
# Compiling the model
cnn_model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
cnn_model.summary()

hist_cnn = cnn_model.fit(
    X_train,
    y_train,
    batch_size=128,
    epochs=15,
    validation_data=(X_val, y_val)
)


scores=cnn_model.evaluate(X_val, y_val)
print("\n%s: %.2f%%" % (cnn_model.metrics_names[1], scores[1]*100))
arreglo = cnn_model.predict(X_val)


# summarize history for accuracy
plt.plot(hist_cnn.history['accuracy'])
plt.plot(hist_cnn.history['val_accuracy'])
plt.title('CNN model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# summarize history for loss
plt.plot(hist_cnn.history['loss'])
plt.plot(hist_cnn.history['val_loss'])
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
