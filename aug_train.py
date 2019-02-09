from train import *
import numpy as np
import argparse
from predict import predict
import pickle


def main():
    random.seed(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', type=str, help='Which model to augment training for.', required=True)

    args = parser.parse_args()

    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    (X, y) = all_dataset()

    root = Path('datasets/nonfaces/')

    all_to_add = []
    for img in tqdm(load_directory(root)):
        err = predict(model, img)
        now_to_add = [d.window(img) for d in err]
        all_to_add.extend(now_to_add)

    Xplus = np.array(all_to_add)
    yplus = np.array([0] * Xplus.shape[0])

    Xnew = np.concatenate((X, Xplus))
    ynew = np.concatenate((y, yplus))

    p = Path('datasets/augmented_data.npz')

    np.savez_compressed(p, X=Xnew, y=ynew)


if __name__ == '__main__':
    main()
