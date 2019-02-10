from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
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

def svm_grid_search(X, y):
    grid = GridSearchCV(LinearSVC(),
                        {'C': [1.0, 2.0, 4.0, 8.0]},
                        n_jobs=-1,
                        cv=4,
                        verbose=3)
    grid.fit(X, y)

    model = grid.best_estimator_

    print('Mean SVM CV score of {}'.format(
        np.mean(cross_val_score(model, X, y, cv=10))))

    model = model.fit(X, y)

    with open('models/svm_model.bin', 'wb') as f:
        pickle.dump(model, f)


def ada_boost_grid_search(X, y):
    # grid = GridSearchCV(AdaBoostClassifier(),
    #                     {'n_estimators': [10, 50, 100],
    #                     },
    #                     cv=4,
    #                     n_jobs=-1,
    #                     verbose=3)
    # grid.fit(X, y)
    # model = grid.best_estimator_

    model = AdaBoostClassifier(n_estimators=100)

    # print('Mean AdaBoost CV score of {}'.format(np.mean(cross_val_score(model, X, y, cv=10))))

    model = model.fit(X, y)

    with open('models/ada_model.bin', 'wb') as f:
        pickle.dump(model, f)


def main():
    random.seed(SEED)

    (X, y) = dataset_hog()

    print('We have {} positive examples.'.format(np.count_nonzero(y == 1)))
    print('We have {} negative examples.'.format(np.count_nonzero(y == 0)))

    ada_boost_grid_search(X, y)
    # svm_grid_search(X, y)



if __name__ == '__main__':
    main()
