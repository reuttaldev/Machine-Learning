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
parser.add_argument("--hidden_layer", default=50, type=int, help="Hidden layer size")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--test_size", default=797, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")

def softmax(z):
    z -= np.max(z, axis=1, keepdims=True)
    z_exp = np.exp(z)
    return z_exp / np.sum(z_exp, axis=1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def main(args: argparse.Namespace) -> tuple[tuple[np.ndarray, ...], list[float]]:

    # Load the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)
    X_train, X_test, t_train, t_test = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=RANDOM_SEED)
    generator = np.random.RandomState(RANDOM_SEED)
    # Generate initial model weights.
    # two matrices. at index 0  n=featues amount x #hidden layer nodes. This are the weights for the input layer. each feature is connected to all nodes in the hidden layer
    # at index 1:  #hidden layer nodes x k classes. These are the weights connecting the nodes of the hidden layer to the output layer (k output nodes, one for each class)
    weights = [generator.uniform(size=[X_train.shape[1], args.hidden_layer], low=-0.1, high=0.1),
               generator.uniform(size=[args.hidden_layer, args.classes], low=-0.1, high=0.1)]
    biases = [np.zeros(args.hidden_layer), np.zeros(args.classes)]

    def forward(inputs):
        # Implementing forward propagation, returning *both* the value of the hidden
        # layer and the value of the output layer.
        # We assume a neural network with a single hidden layer of size `args.hidden_layer`
        # and ReLU activation, where $ReLU(x) = max(x, 0)$, and an output layer with softmax
        # activation.
        # The value of the hidden layer is computed as `ReLU(inputs @ weights[0] + biases[0])`.
        z = inputs @ weights[0]
        h = relu(z+biases[0])
        # The value of the output layer is computed as `softmax(hidden_layer @ weights[1] + biases[1])`.
        y = softmax(h @ weights[1] + biases[1])
        return h,y

    for epoch in range(args.epochs):
        permutation = generator.permutation(X_train.shape[0])
        for batch_count in range(0, len(permutation), args.batch_size):
            batch_indices = permutation[batch_count : batch_count+args.batch_size]
            X_batch = X_train[batch_indices]
            t_batch = t_train[batch_indices]
            h, y = forward(X_batch) 
            t_batch_hotcode = np.eye(args.classes, dtype=int)[np.asarray(t_batch, dtype=int)]
            # Compute the derivative using the chain rule of derivatives:
            # - compute the derivative of the loss with respect to *inputs* of the
            #   softmax on the last layer,
            loss_derivative = y - t_batch_hotcode
            # - compute the derivative with respect to `weights[1]` and `biases[1]`,
            output_weights_derivative = h.T @ loss_derivative 
            # - compute the derivative with respect to the hidden layer output,
            output_biases_derivative = np.sum(loss_derivative, axis=0)  
            # - compute the derivative with respect to the hidden layer input,
            hidden_loss_derivative = (loss_derivative @ weights[1].T) * (h > 0)  
            # - compute the derivative with respect to `weights[0]` and `biases[0]`.
            hidden_weights_derivative = X_batch.T @ hidden_loss_derivative 
            hidden_biases_derivatives = np.sum(hidden_loss_derivative, axis=0)

            weights[1] -= args.learning_rate * output_weights_derivative / args.batch_size 
            biases[1] -= args.learning_rate * output_biases_derivative / args.batch_size 
            weights[0] -= args.learning_rate * hidden_weights_derivative / args.batch_size 
            biases[0] -= args.learning_rate * hidden_biases_derivatives / args.batch_size 

        train_accuracy = np.mean(np.argmax(forward(X_train)[1], axis=1) == t_train)
        test_accuracy = np.mean(np.argmax(forward(X_test)[1], axis=1) == t_test)

        print("After epoch {}: train acc {:.1f}%, test acc {:.1f}%".format(
            epoch + 1, 100 * train_accuracy, 100 * test_accuracy))

    return tuple(weights + biases), [100 * train_accuracy, 100 * test_accuracy]


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    parameters, metrics = main(main_args)
    print("Learned parameters:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in ws.ravel()[:12]] + ["..."]) for ws in parameters), sep="\n")
