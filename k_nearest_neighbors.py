import argparse
import os
import sys
import urllib.request
import numpy as np
import sklearn.metrics
import sklearn.model_selection
import sys
sys.path.append(".")
from config import RANDOM_SEED

parser = argparse.ArgumentParser()
parser.add_argument("--k", default=1, type=int, help="K nearest neighbors to consider")
parser.add_argument("--p", default=2, type=int, help="Use L_p as distance metric")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--test_size", default=500, type=int, help="Test set size")
parser.add_argument("--train_size", default=1000, type=int, help="Train set size")
parser.add_argument("--weights", default="uniform", choices=["uniform", "inverse", "softmax"], help="Weighting to use")

def lp_distance(a, b, p):
    return np.sum(np.abs(a - b) ** p) ** (1 / p)
def softmax(z):
    # To avoid overflow use the fact that     # $softmax(z) = softmax(z + any_constant)$ and compute $softmax(z) = softmax(z - maximum_of_z)$.
    #The key point here is that adding (or subtracting) a constant to each element of z does not change the result of softmax.
    z -= np.max(z)
    # normalize by sum of exponentials
    z_exp = np.exp(z)
    return z_exp / np.sum(z_exp)  # normalize by sum of exponentials

class MNIST:
    """MNIST Dataset.

    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in the range 0-255, the targets are numbers 0-9.
    """
    def __init__(self,
                 name="mnist.train.npz",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset, i.e., `data` and optionally `target`.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value[:data_size])
        self.data = self.data.reshape([-1, 28*28]).astype(float)


def main(args: argparse.Namespace) -> float:
    # Loading MNIST data, scale it to [0, 1] and split it to train and test.
    mnist = MNIST(data_size=args.train_size + args.test_size)
    X_train, X_test, t_train, t_test = sklearn.model_selection.train_test_split(
        mnist.data, mnist.target, test_size=args.test_size, random_state=RANDOM_SEED)

    # Finding the `args.k` nearest neighbors, and use their most frequent target class as prediction,
    # choosing the one with the smallest class number when there are multiple classes with the same frequency.
    # Using $L^p$ norm for a given `args.p` (either 1, 2, or 3) to measure distances.
    
    # The weighting can be:
    # - "uniform": all nearest neighbors have the same weight,
    # - "inverse": `1/distances` is used as weights,
    # - "softmax": `softmax(-distances)` is used as weights.
    test_predictions = []
    test_neighbors = []

    for x_i in X_test:
        # find the closest training examples to this test input
        distances = np.array([lp_distance(x_i,x_j, args.p) for x_j in X_train])
        sorted_indices = np.argsort(distances) # sorts in ascending 
        nearest_neighb_indices = sorted_indices[:args.k]
        nearest_training_targets = t_train[nearest_neighb_indices]
        neighbor_distances = distances[nearest_neighb_indices]

        if args.weights == "softmax":
            weights = softmax(-neighbor_distances) 
        elif args.weights == "uniform":
            weights = np.ones(len(neighbor_distances))
        elif args.weights == "inverse":
            weights = 1 / neighbor_distances 

        target_votes = {}
        for target, weight in zip(nearest_training_targets, weights):
            if target  in target_votes:
                target_votes[target] += weight
            else:
                target_votes[target] = weight
        max_vote_value = max(target_votes.values())
        max_voted_targets=  [t for t, votes in target_votes.items() if votes == max_vote_value]
        voted_t = min(max_voted_targets)
        test_predictions.append(voted_t)
        test_neighbors.append(nearest_neighb_indices)

    accuracy = sklearn.metrics.accuracy_score(t_test, test_predictions)

    if args.plot:
        import matplotlib.pyplot as plt
        examples = [[] for _ in range(10)]
        for i in range(len(test_predictions)):
            if test_predictions[i] != t_test[i] and not examples[t_test[i]]:
                examples[t_test[i]] = [X_test[i], *X_train[test_neighbors[i]]]
        examples = [[img.reshape(28, 28) for img in example] for example in examples if example]
        examples = [[example[0]] + [np.zeros_like(example[0])] + example[1:] for example in examples]
        plt.imshow(np.concatenate([np.concatenate(example, axis=1) for example in examples], axis=0), cmap="gray")
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return 100 * accuracy


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    accuracy = main(main_args)
    print("K-nn accuracy for {} nearest neighbors, L_{} metric, {} weights: {:.2f}%".format(
        main_args.k, main_args.p, main_args.weights, accuracy))
