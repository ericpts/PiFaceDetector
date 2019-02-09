#!/usr/bin/env python3

import requests

def get_faces(url: str, img):
    r = requests.get(url, files={'image': img})
    print(r.text)

def main():
    url = 'http://127.0.0.1:5000/face/'
    with open('/home/ericpts/Downloads/anca.jpg', 'rb') as img:
        get_faces(url, img)

if __name__ == '__main__':
    main()
