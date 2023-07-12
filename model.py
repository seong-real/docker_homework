import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import PIL
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from tensorflow import keras
from keras import layers



class_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

train_dir = '100_1'

X_train = []
y_train = []

for i in range(10):
    for j in range(100):
        train_data = os.listdir(os.path.join(train_dir, class_name[i]))
        f = os.path.join(train_dir, class_name[i], train_data[j])
        image = PIL.Image.open(f)
        arr = np.array(image)
        X_train.append(arr)
        y_train.append(i)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_train = X_train/255.

X_train = X_train.reshape(-1,28, 28, 1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0, stratify=y_train)


y_train_o = to_categorical(y_train)
y_val_o = to_categorical(y_val)

model = keras.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3,3), activation = 'relu', input_shape=(28, 28, 1)),
    layers.MaxPool2D(pool_size=2),
    layers.Conv2D(filters=64, kernel_size=(3,3), activation = 'relu'),
    layers.MaxPool2D(pool_size=2),
    layers.Flatten(),
    layers.Dense(units=32, activation='relu'),
    layers.Dense(units=10, activation='softmax'
    )
])

model.compile(
    optimizer ='adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

EPOCHS = 50
BATCH_SIZE = 256

history = model.fit(
    X_train, y_train_o,
    batch_size = BATCH_SIZE,
    epochs = EPOCHS,
    validation_data=(X_val, y_val_o),
    verbose = 1
)

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch']=history.epoch

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'], hist['loss'], label='Train Loss')
    plt.plot(hist['epoch'], hist['val_loss'], label='Val Loss')

    plt.subplot(1, 2, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(hist['epoch'], hist['accuracy'], label='Train Accuracy')
    plt.plot(hist['epoch'], hist['val_accuracy'], label='Val Accuracy')

    plt.legend()
    plt.show()

plot_history(history)
  