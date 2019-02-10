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
from keras.utils import to_categorical

SEED = 0
HEIGHT = 62
WIDTH = 47

PATCH_SIZE = (HEIGHT, WIDTH)

NPATCHES = 50
SCALES = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


def resize(img):
    return transform.resize(
        img, (HEIGHT, WIDTH), mode='reflect', anti_aliasing=True)


def random_rotation(img):
    random_degree = random.uniform(-25, 25)
    return transform.rotate(img, random_degree)


def random_noise(img):
    return sk.util.random_noise(img)


def horizontal_flip(img):
    return img[:, ::-1]


def random_transformations(img):
    img = sk.img_as_float64(img)
    return [
        img,
        # horizontal_flip(img),
        # random_noise(img),
        # random_noise(img),
        # random_rotation(img),
        # random_rotation(img),
    ]


def plot_random_images(X, fname):
    nimages = X.shape[0]
    n = 1
    while n * n <= nimages and n < 6:
        n += 1
    n -= 1

    fig, ax = plt.subplots(3, 3, figsize=PATCH_SIZE)

    used = {}

    for i, axi in enumerate(ax.flat):
        if len(used) == nimages:
            break

        idx = random.randint(0, nimages - 1)
        while idx in used:
            idx = random.randint(0, nimages - 1)

        used[idx] = True
        axi.imshow(X[idx], cmap='gray')
        axi.axis('off')
    plt.savefig(fname)


def extract_patches(img, npatches, scale):
    patch_size = (HEIGHT, WIDTH)
    extracted_patch_size = tuple((scale * np.array(patch_size)).astype(int))

    extractor = PatchExtractor(
        patch_size=extracted_patch_size,
        max_patches=npatches,
        random_state=SEED)
    try:
        patches = extractor.transform(img[np.newaxis])
        if scale != 1.0:
            patches = np.array([resize(patch) for patch in patches])
        return patches
    except ValueError:
        return []


def dataset_cache(dataset_name):
    fname = 'datasets/{}_data.npz'.format(dataset_name)

    def decorator(fn):

        def fn_wrapper():
            p = Path(fname)

            if p.exists():
                dset = np.load(p)
                X = dset['X']
                y = dset['y']
            else:
                (X, y) = fn()

            assert X.dtype == np.float64
            assert X.shape[0] == y.shape[0]

            if not p.exists():
                np.savez_compressed(p, X=X, y=y)

            return (X, y)

        return fn_wrapper

    return decorator


def load_sklearn():
    neg_imgs_to_use = [
        'camera', 'text', 'coins', 'moon', 'page', 'clock',
        'immunohistochemistry', 'chelsea', 'coffee', 'hubble_deep_field'
    ]
    imgs = [color.rgb2gray(getattr(data, name)()) for name in neg_imgs_to_use]

    patches = [
        extract_patches(im, NPATCHES, scale) for im in imgs for scale in SCALES
    ]
    patches = [p for p in patches if len(p) > 0]

    X = np.vstack(patches)
    nimages = X.shape[0]
    y = np.array([0] * nimages)

    plot_random_images(X, "sklearn_fig.png")

    return (X, y)


def load_directory(root):
    images = []
    for p in tqdm(
            itertools.chain(
                root.glob('**/*.jpg'),
                root.glob('**/*.png'),
            )):
        img = sk.io.imread(p)
        img = color.rgb2gray(img)
        images.append(img)

    return images


def load_positive_directory():
    print('Loading the positive directory')

    root = Path('datasets/faces/')
    images = load_directory(root)
    X = np.array([resize(img) for img in images])
    y = np.array([1] * X.shape[0])

    plot_random_images(X, "positive_directory_fig.png")

    return (X, y)


def load_negative_directory():
    print('Loading the negative directory')

    root = Path('datasets/nonfaces/')
    images = load_directory(root)

    X = []
    for img in images:
        if img.shape > 2 * PATCH_SIZE:
            patches = [
                extract_patches(img, NPATCHES, scale) for scale in SCALES
            ]

            for p in patches:
                X.extend(p)
        else:
            x.append(resize(img))

    X = np.stack(X)
    y = np.array([0] * X.shape[0])

    plot_random_images(X, "negative_directory_fig.png")

    return (X, y)


@dataset_cache("positive")
def positive_dataset():
    X = []
    y = []

    for dset in [load_positive_directory]:
        (X_now, y_now) = dset()
        X.extend(X_now)
        y.extend(y_now)

    X = np.array(X)
    y = np.array(y)

    assert X.dtype == np.float64
    assert X.shape[0] == y.shape[0]

    return (X, y)


@dataset_cache("negative")
def negative_dataset():
    X = []
    y = []
    for dset in [load_sklearn, load_negative_directory]:
        (X_now, y_now) = dset()
        X.extend(X_now)
        y.extend(y_now)

    X = np.stack(X)
    y = np.stack(y)

    assert X.shape[0] == y.shape[0]
    return X, y


@dataset_cache("dataset_no_hog")
def dataset_no_hog():
    X_pos, y_pos = positive_dataset()
    X_neg, y_neg = negative_dataset()

    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((y_pos, y_neg))

    X = np.reshape(X, (-1, HEIGHT, WIDTH, 1))
    y = to_categorical(y)

    return (X, y)


@dataset_cache("hog_dataset")
def dataset_hog():
    X_pos, y_pos = positive_dataset()
    X_neg, y_neg = negative_dataset()

    assert (X_pos.shape[0] == y_pos.shape[0])
    assert (X_neg.shape[0] == y_neg.shape[0])

    X = [feature.hog(img) for img in itertools.chain(X_pos, X_neg)]
    y = np.concatenate((y_pos, y_neg))

    X = np.array(X)

    assert X.dtype == np.float64

    return (X, y)

