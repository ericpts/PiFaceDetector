from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.model_selection import GridSearchCV
import random
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.image import PatchExtractor
from skimage import data, transform, color, feature
import skimage as sk
from pathlib import Path
import random
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
import pickle
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from load_dataset import *
import pickle

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('datasets/data/' + ID + '.npz')['arr_0']

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


def identitiy(f, k):
    def fn(input):
        [f1, f2, f3] = f

        X = keras.layers.Conv2D(f1, kernel_size=1)(input)
        X = keras.layers.BatchNormalization(axis=3)(X)
        X = keras.layers.Activation('relu')(X)

        X = keras.layers.Conv2D(f2, kernel_size=k, padding='same')(X)
        X = keras.layers.BatchNormalization(axis=3)(X)
        X = keras.layers.Activation('relu')(X)

        X = keras.layers.Conv2D(f3, kernel_size=1)(X)
        X = keras.layers.BatchNormalization(axis=3)(X)

        X_input = input

        X = keras.layers.Add()([X, X_input])
        X = keras.layers.Activation('relu')(X)
        return X
    return fn


def convolutional(f, k, s):
    def fn(input):
        [f1, f2, f3] = f

        X = keras.layers.Conv2D(f1, kernel_size=1, strides=(s, s))(input)
        X = keras.layers.BatchNormalization(axis=3)(X)
        X = keras.layers.Activation('relu')(X)

        X = keras.layers.Conv2D(f2, kernel_size=k, padding='same')(X)
        X = keras.layers.BatchNormalization(axis=3)(X)
        X = keras.layers.Activation('relu')(X)

        X = keras.layers.Conv2D(f3, kernel_size=1)(X)
        X = keras.layers.BatchNormalization(axis=3)(X)

        X_input = keras.layers.Conv2D(f3, kernel_size=1, strides=(s, s))(input)
        X_input = keras.layers.BatchNormalization(axis=3)(X_input)

        X = keras.layers.Add()([X, X_input])
        X = keras.layers.Activation('relu')(X)
        return X
    return fn


def main():

    params = {'dim': (HEIGHT,WIDTH),
          'batch_size': 128,
          'n_classes': 2,
          'n_channels': 1,
          'shuffle': True}

    root = Path('datasets/data')

    with open(root / 'labels.txt', 'rb') as f:
        labels = pickle.load(f)

    all_ids = list(labels.keys())
    random.shuffle(all_ids)

    percentage_validation = 0.20

    val_length = int(percentage_validation * len(all_ids))
    train_length = len(all_ids) - val_length

    partition = {
            'train': all_ids[: train_length],
            'validation': all_ids[train_length: ],
    }

    training_generator = DataGenerator(partition['train'], labels, **params)
    validation_generator = DataGenerator(partition['validation'], labels, **params)


    input = keras.layers.Input(shape=(HEIGHT, WIDTH, 1))

    X = keras.layers.Conv2D(64, kernel_size=7, strides=2)(input)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.Activation('relu')(X)

    X = convolutional([64, 64, 32], 3, 1)(X)
    X = identitiy([64, 64, 32], 3)(X)
    X = identitiy([64, 64, 32], 3)(X)

    flat = keras.layers.Flatten()(X)

    predictions = Dense(2, activation='softmax')(flat)

    model = keras.models.Model(inputs=input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    model.fit_generator(
            generator=training_generator,
            validation_data=validation_generator,
            use_multiprocessing=True,
            workers=2,
            epochs=3
            )


    with open('models/nn_model.bin', 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    main()
