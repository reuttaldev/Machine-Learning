import argparse
import numpy as np
import sklearn.datasets
import sklearn.model_selection
import sys
sys.path.append(".")
from config import RANDOM_SEED

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--epochs", default=50, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def get_metrics(X,t,weights, threshold):
    prob_predictions = sigmoid(X @ weights) # same as doing np.dot. it returns a single prob (of it being in the first class) per input
    
    #accuracy 
    cat_predictions = [1 if i > threshold else 0 for i in prob_predictions]
    # find the number of correct predictions
    correct_cat_count = np.sum(cat_predictions == t)
    accuracy = correct_cat_count / len(t)

    # the loss 
    neg_log_loss = -np.mean(t * np.log(prob_predictions) + (1 - t) * np.log(1 - prob_predictions))
    return accuracy,neg_log_loss

def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
    # Generating an artificial classification dataset.
    X, t = sklearn.datasets.make_classification(
        n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0, random_state=RANDOM_SEED)

    # Addding constant feature with value 1 to represent the bias
    X = np.concatenate((X, np.ones(shape= (X.shape[0],1))), axis = 1)
    X_train, X_test, t_train, t_test = sklearn.model_selection.train_test_split(X,t,test_size=args.test_size, random_state=RANDOM_SEED)
    # Generating initial logistic regression weights.
    generator = np.random.RandomState(RANDOM_SEED)
    weights = generator.uniform(size=X_train.shape[1], low=-0.1, high=0.1)

    for epoch in range(args.epochs):
        # Random order to process the data points in
        permutation = generator.permutation(X_train.shape[0])
        for batch_count in range(0, len(permutation), args.batch_size):
            batch_indices = permutation[batch_count : batch_count+args.batch_size]
            X_batch = X_train[batch_indices]
            t_batch = t_train[batch_indices]

            batch_gradient = np.zeros(weights.shape)
            for x_i,t_i in zip(X_batch, t_batch):
                # Same loss as in linear regression, but now the sigmoind function is applied on the prediction
                # to represent the probability of this instance being category 1. 
                prediction = sigmoid(np.dot(x_i, weights))
                error = prediction - t_i
                loss_gradient = error * x_i
                batch_gradient += loss_gradient

            batch_gradient /= args.batch_size
            # Update the weights accordinaly -- the SGD update is weights = weights - args.learning_rate * gradient
            weights -= args.learning_rate * batch_gradient

        # After the SGD epoch, I measure the average loss and accuracy for both the
        # train set and the test set. The loss is the average MLE loss (i.e., the
        # negative log-likelihood, or cross-entropy loss, or KL loss) per example.

        train_accuracy, train_loss = get_metrics(X_train,t_train, weights, 0.5) 
        test_accuracy, test_loss = get_metrics(X_test, t_test, weights, 0.5) 
        print("After epoch {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
            epoch + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))

        if args.plot:
            import matplotlib.pyplot as plt
            if args.plot is not True:
                plt.gcf().get_axes() or plt.figure(figsize=(6.4*3, 4.8*(args.epochs+2)//3))
                plt.subplot(3, (args.epochs+2)//3, 1 + epoch)
            xs = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 50)
            ys = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 50)
            predictions = [[1 / (1 + np.exp(-([x, y, 1] @ weights))) for x in xs] for y in ys]
            plt.contourf(xs, ys, predictions, levels=20, cmap="RdBu", alpha=0.7)
            plt.contour(xs, ys, predictions, levels=[0.25, 0.5, 0.75], colors="k")
            plt.scatter(X_train[:, 0], X_train[:, 1], c=t_train, label="train", marker="P", cmap="RdBu")
            plt.scatter(X_test[:, 0], X_test[:, 1], c=t_test, label="test", cmap="RdBu")
            plt.legend(loc="upper right")
            plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return weights, [(train_loss, 100 * train_accuracy), (test_loss, 100 * test_accuracy)]


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(main_args)
    print("Learned weights", *("{:.2f}".format(weight) for weight in weights))
