#!/usr/bin/env python3
import sklearn.feature_extraction as fextr
import matplotlib.pyplot as plt
import skimage as sk
import pickle
import argparse
import numpy as np
import time
from typing import List

from train import HEIGHT, WIDTH, plot_random_images

OVERLAP_THRESH = 0.25

class Detection(object):

    def __init__(self, x: int, y: int, h: int, w: int, score: float):
        self.x = x
        self.y = y
        self.h = h
        self.w = w
        self.score = score

    @property
    def x1(self):
        return self.x

    @property
    def x2(self):
        return self.x + self.h

    @property
    def y1(self):
        return self.y

    @property
    def y2(self):
        return self.y + self.w

    def window(self, img):
        return img[self.x1 : self.x2, self.y1 : self.y2]


def for_scale(model, img, scale) -> List[Detection]:
    patch_size = (HEIGHT, WIDTH)

    scale_factor = 1

    while (np.prod(img.shape) > np.prod(patch_size) * scale):
        img = sk.transform.rescale(
            img, 0.8, multichannel=False, mode='reflect', anti_aliasing=True)
        scale_factor *= 0.8

    scaled_h = int(HEIGHT / scale_factor)
    scaled_w = int(WIDTH / scale_factor)
    (h, w) = (img.shape)

    if h < HEIGHT or w < WIDTH:
        return []

    ret = []
    def add_detection(i, j, score):
        orig_x = int(i / scale_factor)
        orig_y = int(j / scale_factor)
        ret.append(Detection(orig_x, orig_y, scaled_h, scaled_w, score))

    def score_window(window):
        X = np.reshape(sk.feature.hog(window, block_norm='L1'), (1, -1))
        prob = model.predict_proba(X)[0][1]
        return prob

    def maybe_add_window(i, j, window):
        score = score_window(window)
        if score > 0.52:
            print(score)
            add_detection(i, j, score)


    def new():
        shape = (h - HEIGHT + 1, w - WIDTH + 1, HEIGHT, WIDTH)
        strides = 2 * img.strides

        patches = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
        patches = patches.reshape(-1, HEIGHT, WIDTH)

        output_img = np.array([score_window(roi) for roi in patches])
        output_img.reshape((h - HEIGHT + 1, w - WIDTH + 1))


    def old():
        windows = []

        for i in range(0, h - HEIGHT):
            for j in range(0, w - WIDTH):
                window = img[i:i + HEIGHT, j:j + WIDTH]
                windows.append(window)

        [maybe_add_window(i, j, window) for window in windows]

    print('Doing for_scale()')
    print('\tscale = {}'.format(scale))
    print('\tshape = {}'.format(img.shape))
    t = time.time()

    new()

    print('Done after {}'.format(time.time() - t))

    return ret


def filter_duplicates(detections: List[Detection]) -> List[Detection]:
    pick = []
    x1 = [d.x1 for d in detections]
    x2 = [d.x2 for d in detections]
    y1 = [d.y1 for d in detections]
    y2 = [d.y2 for d in detections]

    x1 = np.array(x1)
    x2 = np.array(x2)
    y1 = np.array(y1)
    y2 = np.array(y2)

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        for pos in range(0, last):
            j = idxs[pos]

            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            overlap = float(w * h) / area[j]

            if overlap > OVERLAP_THRESH:
                suppress.append(pos)

        idxs = np.delete(idxs, suppress)

    ret = [detections[i] for i in pick]
    return ret


def predict(model, img):
    original_img = img

    detections = []
    for scale in [1, 2, 4, 8, 16, 32]:
        detections.extend(for_scale(model, original_img, scale))

    print('Found {} raw detection'.format(len(detections)))

    detections = filter_duplicates(detections)

    print('Found {} unique detection'.format(len(detections)))

    return detections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image', type=str, help='Image to predict for.', required=True)
    parser.add_argument(
        '--model', type=str, help='Which model to augment training for.', required=True)

    args = parser.parse_args()

    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    img_path = args.image
    img = sk.io.imread(img_path)
    img = sk.color.rgb2gray(img)

    faces = predict(model, img)
    patches = [f.window(img) for f in faces]
    plot_random_images(np.array(patches), "fig.png")


if __name__ == '__main__':
    main()
