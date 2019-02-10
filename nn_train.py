from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.model_selection import GridSearchCV
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

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

def keras_model(X, y):
    model = Sequential()

    X = np.reshape(X, (-1, HEIGHT, WIDTH, 1))
    y = to_categorical(y)

    model.add(Conv2D(64, kernel_size=3, activation='relu',
        input_shape=(HEIGHT, WIDTH, 1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy',
            metrics=['accuracy'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


    print(X_train.shape)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

    with open('nn_model.bin', 'wb') as f:
        pickle.dump(model, f)

def main():
    random.seed(SEED)

    (X, y) = dataset_no_hog()

    print('We have {} positive examples.'.format(np.count_nonzero(y == 1)))
    print('We have {} negative examples.'.format(np.count_nonzero(y == 0)))

    keras_model(X, y)



if __name__ == '__main__':
    main()
