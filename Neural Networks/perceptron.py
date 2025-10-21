import argparse

import numpy as np
import sklearn.datasets

import sys
sys.path.append(".")
from config import RANDOM_SEED

parser = argparse.ArgumentParser()
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")

def main(args: argparse.Namespace) -> np.ndarray:
    generator = np.random.RandomState(RANDOM_SEED)

    # Generating binary classification data with labels {-1, 1}.
    X_train, t_train = sklearn.datasets.make_classification(
        n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0,
        n_clusters_per_class=1, flip_y=0, class_sep=2, random_state=RANDOM_SEED)
    t_train = 2 * t_train - 1

    X_train = np.concatenate((X_train, np.ones(shape= (X_train.shape[0],1))), axis = 1)
    # Generating initial perceptron weights.
    weights = np.zeros(X_train.shape[1])

    done = False
    # This is basically epoch, but it stops when we find weights that separetes the data 
    while not done:
        permutation = generator.permutation(X_train.shape[0])
        all_correct = True
        for i in permutation:
            x_i = X_train[i] 
            # Finding if x_i is classified correctly with the current weights
            t_i = t_train[i]
            # x_i and w both have the same shape, the number of columns (features) in X. ti is 1 or -1
            classified_correctly = t_i * np.dot(x_i.T ,weights) >0
            if classified_correctly:
                continue
            # it is not - update the weights
            weights+= t_i * x_i
            all_correct = False

        # If all training instances are correctly classified, set `done=True`, otherwise set `done=False`.
        if all_correct:
            done = True

        if args.plot and not done:
            import matplotlib.pyplot as plt
            if args.plot is not True:
                plt.gcf().get_axes() or plt.figure(figsize=(6.4*3, 4.8*3))
                plt.subplot(3, 3, 1 + len(plt.gcf().get_axes()))
            plt.scatter(X_train[:, 0], X_train[:, 1], c=t_train)
            xs = np.linspace(*plt.gca().get_xbound() + (50,))
            ys = np.linspace(*plt.gca().get_ybound() + (50,))
            plt.contour(xs, ys, [[[x, y, 1] @ weights for x in xs] for y in ys], levels=[0])
            plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return weights

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    weights = main(main_args)
    print("Learned weights", *("{:.2f}".format(weight) for weight in weights))
