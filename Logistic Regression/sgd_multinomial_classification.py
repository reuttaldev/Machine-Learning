import argparse
import numpy as np
import sklearn.datasets
import sklearn.model_selection

import sys
sys.path.append(".")
from config import RANDOM_SEED

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--test_size", default=797, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")

def softmax(z):
    # Note that you need to be careful when computing softmax because the exponentiation
    # TO avoid overflow, I use the fact that $softmax(z) = softmax(z + any_constant)$ and compute $softmax(z) = softmax(z - maximum_of_z)$.
    # That way only non-positive values get exponentiated, and overflow does not occur. 
    # The key point here is that adding (or subtracting) a constant to each element of z does not change the result of softmax.
    z -= np.max(z)
    # normalize by sum of exponentials
    z_exp = np.exp(z)
    return z_exp / np.sum(z_exp)  # normalize by sum of exponentials

def get_metrics(X,t,weights):
    # Doing it on each vector in the matrix and not on the matrix inself bc the sum in the softmax will be different, I think
    prob_predictions =np.array([softmax(y_i) for y_i in (X @ weights)])
    # Getting the class with the highest prob 
    cat_predictions = np.argmax(prob_predictions,axis=1)
    # Find the number of correct predictions
    correct_cat_sum = np.sum(cat_predictions == t)
    accuracy = correct_cat_sum / len(t)

    # The formula for negative log likelihood is the negative mean of the sum of P(X)log(Q(X)) where P is the actual value and Q is our prediction
    # The actual probability P is 1 for the target class and 0 for everything else. So only do the formula for the classes where it won't be 0, then it's just log(Q(X)) 
    n=prob_predictions.shape[0]
    #does the same as     correct_class_probs = [row[i] for row, i in zip(prob_predictions, t)]
    correct_class_probs = prob_predictions[np.arange(n), t]
    neg_log_loss = - np.sum(np.log(correct_class_probs))
    neg_log_loss_avg = neg_log_loss / n
    #neg_log_loss_avg = sklearn.metrics.log_loss(t,prob_predictions)
    return accuracy,neg_log_loss_avg

def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:

    # Load the digits dataset.
    X, t = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)
    # Accounting for bias
    X = np.pad(X, [(0, 0), (0, 1)], constant_values=1)
    X_train, X_test, t_train, t_test = sklearn.model_selection.train_test_split(
        X, t, test_size=args.test_size, random_state=RANDOM_SEED)

    # Generating initial model weights.
    # The weights are a matrix with number of rows = number of input features, number of columns = number of classes
    generator = np.random.RandomState(RANDOM_SEED)
    weights = generator.uniform(size=[X_train.shape[1], args.classes], low=-0.1, high=0.1)

    for epoch in range(args.epochs):
        permutation = generator.permutation(X_train.shape[0])
        for batch_count in range(0, len(permutation), args.batch_size):
            batch_indices = permutation[batch_count : batch_count+args.batch_size]
            X_batch = X_train[batch_indices]
            t_batch = t_train[batch_indices]

            batch_gradient = np.zeros(weights.shape)
            for x_i,t_i in zip(X_batch, t_batch):
                prediction = softmax(np.dot(x_i,weights))
                # ti is the class number (classes are digits 0-9), turn it into one hot encoding
                t_i = [1 if t_i == i else 0 for i in range(args.classes)]
                # pridiction is a list of prob. target is 1 hot encoding. both a vector with k entries when k is the number of classes
                error = prediction - t_i
                # xi = vector where n = number of features. error = vector where k = number of classes
                # make it so x_i is of shape (n,1) and error is of size (1,k), and when we multiply them we 
                # will get a matrix (n,k) which is the size of the weight matrix 
                error = error .reshape(1, -1) # making it 2d and making the rows the columns
                x_i = x_i.reshape(-1,1) # just making it 2d
                gradient =  np.dot(x_i,error)
                batch_gradient += gradient

            batch_gradient /= args.batch_size
            # Update the weights accordinaly -- the SGD update is weights = weights - args.learning_rate * gradient
            weights -= args.learning_rate * batch_gradient

        train_accuracy, train_loss = get_metrics(X_train,t_train, weights) 
        test_accuracy, test_loss = get_metrics(X_test, t_test, weights) 

        print("After epoch {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
            epoch + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))

    return weights, [(train_loss, 100 * train_accuracy), (test_loss, 100 * test_accuracy)]


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(main_args)
    print("Learned weights:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")
