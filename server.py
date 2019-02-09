#!/usr/bin/env python3
from flask import Flask, request
import skimage as sk
import pickle

from predict import predict

app = Flask("Electric Eye")

with open('model.bin', 'rb') as f:
    model = pickle.load(f)


@app.route("/face/")
def detect_faces():
    img = request.files['image']
    img = sk.io.imread(img, as_gray=True)

    faces = predict(model, img)

    print(faces)

    return "Hit /face/"
