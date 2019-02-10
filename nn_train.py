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


def main():

    params = {'dim': (HEIGHT,WIDTH),
          'batch_size': 64,
          'n_classes': 2,
          'n_channels': 1,
          'shuffle': True}

    root = Path('datasets/data')

    with open(root / 'labels.txt', 'rb') as f:
        labels = pickle.load(f)

    all_ids = list(labels.keys())
    random.shuffle(all_ids)

    percentage_validation = 0.25

    val_length = int(percentage_validation * len(all_ids))
    train_length = len(all_ids) - val_length

    partition = {
            'train': all_ids[: train_length],
            'validation': all_ids[train_length: ],
    }

    training_generator = DataGenerator(partition['train'], labels, **params)
    validation_generator = DataGenerator(partition['validation'], labels, **params)


    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation='relu',
        input_shape=(HEIGHT, WIDTH, 1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy',
            metrics=['accuracy'])


    model.fit_generator(
            generator=training_generator,
            validation_data=validation_generator,
            use_multiprocessing=True,
            workers=2
            )


    with open('models/nn_model.bin', 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    main()
