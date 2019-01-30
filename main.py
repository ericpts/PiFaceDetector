from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.image import PatchExtractor
from skimage import data, transform, color, feature
import skimage as sk
from pathlib import Path
import random
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cifar10
import itertools
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB


SEED = 0
HEIGHT = 62
WIDTH = 47
NPATCHES = 500
SCALES = [0.5, 1.0, 2.0]

def random_rotation(img):
    random_degree = random.uniform(-25, 25)
    return transform.rotate(img, random_degree)


def random_noise(img):
    return sk.util.random_noise(img)


def horizontal_flip(img):
    return img[:, ::-1]


def random_transformations(img):
    return [
        img,
        horizontal_flip(img),
        random_noise(img),
        random_noise(img),
        random_rotation(img),
        random_rotation(img),
        ]


def plot_random_images(X, fname):
    fig, ax = plt.subplots(6, 10)
    for i, axi in enumerate(ax.flat):
        axi.imshow(X[random.randint(0, nimages - 1)], cmap='gray')
        axi.axis('off')
    plt.savefig(fname)


def load_lfw():
    lfw_people = fetch_lfw_people()

    n_samples, h, w = lfw_people.images.shape

    assert (h, w) == (HEIGHT, WIDTH)

    X = np.reshape(lfw_people.data, (n_samples, HEIGHT, WIDTH))

    # Now do some augmentation.
    X = np.concatenate(
        [random_transformations(img) for img in X]
    )

    y = np.array([1] * X.shape[0])

    assert (X.shape[0] == y.shape[0])

    return (X, y)

def extract_patches(img, npatches, scale):
    patch_size = (HEIGHT, WIDTH)
    extracted_patch_size = tuple((scale * np.array(patch_size)).astype(int))
    extractor = PatchExtractor(patch_size = extracted_patch_size, max_patches=npatches, random_state=SEED)
    patches = extractor.transform(img[np.newaxis])
    if scale != 1.0:
        patches = np.array([transform.resize(patch, patch_size, mode='reflect', anti_aliasing=True) for patch in patches])
    return patches

def load_sklearn():
    neg_imgs_to_use = ['camera', 'text', 'coins', 'moon',
                'page', 'clock', 'immunohistochemistry',
                'chelsea', 'coffee', 'hubble_deep_field']
    imgs = [color.rgb2gray(getattr(data, name)())
            for name in neg_imgs_to_use]

    X = np.vstack([extract_patches(im, NPATCHES, scale) for im in imgs for scale in SCALES])
    nimages = X.shape[0]
    y = np.array([0] * nimages)

    return (X, y)

def load_cifar10():
    cifar10.maybe_download_and_extract()

    fst = cifar10.load_training_data()[0]
    snd = cifar10.load_test_data()[0]

    raw_imgs = np.concatenate((fst, snd))
    X = np.array([
        transform.resize(color.rgb2gray(img), (HEIGHT, WIDTH), mode='reflect', anti_aliasing=True) for img in raw_imgs])

    nimages = X.shape[0]
    y = np.array([0] * nimages)

    return (X, y)

def dataset_cache(dataset_name):
    fname = 'datasets/{}_data.bin'.format(dataset_name)
    def decorator(fn):
        def fn_wrapper():
            p = Path(fname)
            if p.exists():
                with p.open('rb') as f:
                    return pickle.load(f)
            else:
                (X, y) = fn()
                with p.open('wb') as f:
                    pickle.dump((X, y), f)
                return (X, y)
        return fn_wrapper
    return decorator


@dataset_cache("imagenet")
def load_imagenet():
    root = Path('datasets/imagenet/valid')

    at = 0
    images = []
    for p in root.glob('**/*.png'):
        img = sk.io.imread(p, as_gray=True)
        img = transform.resize(img, (HEIGHT, WIDTH))
        images.append(img)
        at += 1
        if at % 1000 == 0:
            print('.', end='')

    X = np.concatenate(
        [random_transformations(img) for img in images]
    )

    y = np.array([1] * X.shape[0])

    return (X, y)


@dataset_cache("yale")
def load_yale():
    root = Path('datasets/CroppedYale/')

    images = []
    for p in root.glob('**/*.pgm'):
        img = sk.io.imread(p, as_gray=True)
        img = transform.resize(img, (HEIGHT, WIDTH))
        images.append(img)

    X = np.concatenate(
        [random_transformations(img) for img in images]
    )

    y = np.array([1] * X.shape[0])

    return (X, y)


@dataset_cache("positive")
def positive_dataset():
    X = []
    y = []

    for dset in [load_lfw, load_yale]:
        (X_now, y_now) = dset()
        X.extend(X_now)
        y.extend(y_now)

    X = np.array(X)
    y = np.array(y)

    assert X.shape[0] == y.shape[0]
    return X, y

    return load_lfw()



@dataset_cache("negative")
def negative_dataset():
    X = []
    y = []
    for dset in [load_sklearn, load_cifar10, load_imagenet]:
        (X_now, y_now) = dset()
        X.extend(X_now)
        y.extend(y_now)

    X = np.array(X)
    y = np.array(y)

    assert X.shape[0] == y.shape[0]
    return X, y

@dataset_cache("all_dataset")
def all_dataset():
    X_pos, y_pos = positive_dataset()
    X_neg, y_neg = negative_dataset()

    assert (X_pos.shape[0] == y_pos.shape[0])
    assert (X_neg.shape[0] == y_neg.shape[0])

    X = [feature.hog(img) for img in itertools.chain(X_pos, X_neg)]
    y = np.concatenate((y_pos, y_neg))

    X = np.array(X)

    return (X, y)

def main():
    random.seed(SEED)

    (X, y) = all_dataset()

    print('We have {} positive examples.'.format(np.count_nonzero(y == 1)))
    print('We have {} negative examples.'.format(np.count_nonzero(y == 0)))

    print(cross_val_score(GaussianNB(), X, y, cv=10))


if __name__ == '__main__':
    main()

