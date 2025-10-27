import argparse
import os
import urllib.request
import warnings
import numpy as np
import sklearn.base
import sklearn.exceptions
import sklearn.linear_model
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sys
sys.path.append(".")
from config import RANDOM_SEED
parser = argparse.ArgumentParser()
parser.add_argument("--data_size", default=5000, type=int, help="Data size")
parser.add_argument("--max_iter", default=100, type=int, help="Maximum iterations for LR")
parser.add_argument("--pca", default=None, type=int, help="PCA dimensionality")
parser.add_argument("--solver", default="saga", type=str, help="LR solver")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")

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


class PCATransformer(sklearn.base.TransformerMixin):
    def __init__(self, n_components, seed):
        self._n_components = n_components
        self._seed = seed

    def fit(self, X, y=None):
        generator = np.random.RandomState(RANDOM_SEED)
        X_minus_mean = X - np.mean(X, axis=0)
        S = np.dot(X_minus_mean.T, X_minus_mean) / X.shape[0]
        # Using the power iteration algorithm for <= 10 dimensions.
        if self._n_components <= 10:
            self._V = np.zeros((X.shape[1], self._n_components))
            for i in range(self._n_components):
                v = generator.uniform(-1, 1, size=X.shape[1])
                for _ in range(10):
                    v = np.dot(S, v)
                    v -= np.dot(self._V, np.dot(self._V.T, v))
                    v /= np.linalg.norm(v)
                self._V[:, i] = v

        else:
            _, _, Vt = np.linalg.svd(S, full_matrices=False)
            self._V = Vt.take(range(self._n_components), axis=0).T

        # We round the principal components to avoid rounding errors during
        # ReCodEx evaluation.
        self._V = np.around(self._V, decimals=4)

        return self

    def transform(self, X):
        # TODO: Transform the given `X` using the precomputed `self._V`.
        return np.dot(X, self._V)


def main(args: argparse.Namespace) -> float:
    # Suppress warnings about the solver not converging because we
    # deliberately use a low `max_iter` for the models to train quickly.
    warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)

    # Use the MNIST dataset.
    dataset = MNIST(data_size=args.data_size)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        dataset.data, dataset.target, test_size=args.test_size, random_state= RANDOM_SEED)

    pca = [("PCA", PCATransformer(args.pca, RANDOM_SEED))] if args.pca else []

    pipeline = sklearn.pipeline.Pipeline([
        ("scaling", sklearn.preprocessing.MinMaxScaler()),
        *pca,
        ("classifier", sklearn.linear_model.LogisticRegression(
            solver=args.solver, max_iter=args.max_iter, random_state= RANDOM_SEED)),
    ])
    pipeline.fit(train_data, train_target)

    test_accuracy = pipeline.score(test_data, test_target)
    return test_accuracy


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    accuracy = main(main_args)
    print("Test set accuracy: {:.2f}%".format(100 * accuracy))
