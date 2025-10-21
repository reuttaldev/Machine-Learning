import argparse
import numpy as np
import sklearn.datasets
import sklearn.model_selection
import sys
sys.path.append(".")
from config import RANDOM_SEED

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=5, type=int, help="Number of classes to use")
parser.add_argument("--data_size", default=200, type=int, help="Data size")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X,weights,threshold=0.5):
    # X is of size inputs x features and weights is features x classes. 
    # so the resulting matrix will be inputs x classes which is exactly what we want
    prob_predictions = sigmoid(X @ weights)
    return (prob_predictions >= threshold).astype(np.uint8)

def f1_score(tp,fp,fn):
    d =  (2*tp + fp + fn)
    if d == 0:
        return 0
    return 2*tp / (2*tp + fp + fn)

def get_metrics(y,t):
    # It will be an array with the values per class (it sums over rows)
    y = y.astype(bool)
    t = t.astype(bool)
    tp = np.sum(y & t, axis=0)
    fp = np.sum(y & ~t, axis=0)
    fn = np.sum(~y & t, axis=0)

    # In macro, calculate the F1-score independently for each class and then takes their arithmetic mean.
    # This will be an array where each value is the f1 score for the class at that index
    f1_per_class = [f1_score(tp[c],fp[c],fn[c]) for c in range(y.shape[1])]
    f1_macro = np.mean(f1_per_class)
    # Micro; calculate metrics globally i.e. the tp,fp,fn are shared between all classes
    f1_micro = f1_score(sum(tp),sum(fp),sum(fn))
    return f1_micro,f1_macro

def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
    # Generate an artificial classification dataset.
    X, t = sklearn.datasets.make_multilabel_classification(
        n_samples=args.data_size, n_classes=args.classes, allow_unlabeled=False,
        return_indicator=False, random_state=RANDOM_SEED)
    # Convert t (a list of classes for every input example) to a dense representation (n-hot encoding)
    target = np.zeros((len(t), args.classes), dtype=np.uint8)
    for i,t_list in enumerate(t):
        for c in t_list:
            target[i,c] = 1
    # Account for bias in input features
    X = np.pad(X, [(0, 0), (0, 1)], constant_values=1)

    X_train, X_test, t_train, t_test = sklearn.model_selection.train_test_split(
        X, target, test_size=args.test_size, random_state=RANDOM_SEED)

    # Weights is a 2d matrix with dimentions num of features x classes 
    generator = np.random.RandomState(RANDOM_SEED)
    weights = generator.uniform(size=[X_train.shape[1], args.classes], low=-0.1, high=0.1)

    for epoch in range(args.epochs):
        permutation = generator.permutation(X_train.shape[0])
        for batch_count in range(0, len(permutation), args.batch_size):
            batch_indices = permutation[batch_count : batch_count+args.batch_size]
            X_batch = X_train[batch_indices]
            t_batch = t_train[batch_indices]

            # For multilabel classification I do binary classification for each class per input. 
            batch_gradient = np.zeros(weights.shape)
            #  Can be done on entire batch without the inner loops. this is just for clarity; 
            # to show I am doing exactly the same as I did in binary classification but for several c
            for x_i,t_i in zip(X_batch, t_batch):
                for c in range(args.classes):
                    #:, gets the column. 
                    prediction = sigmoid(np.dot(x_i, weights[:,c]))
                    error = prediction - t_i[c]
                    loss_gradient = error * x_i
                    batch_gradient[:,c] += loss_gradient
            batch_gradient /= args.batch_size
            weights -= args.learning_rate * batch_gradient

        predictions_train = predict(X_train,weights)
        predictions_test = predict(X_test,weights)
        train_f1_micro, train_f1_macro = get_metrics(predictions_train,t_train) 
        test_f1_micro, test_f1_macro = get_metrics(predictions_test,t_test)

        print("After epoch {}: train F1 micro {:.2f}% macro {:.2f}%, test F1 micro {:.2f}% macro {:.1f}%".format(
            epoch + 1, 100 * train_f1_micro, 100 * train_f1_macro, 100 * test_f1_micro, 100 * test_f1_macro))

    return weights, [(100 * train_f1_micro, 100 * train_f1_macro), (100 * test_f1_micro, 100 * test_f1_macro)]


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(main_args)
    print("Learned weights:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")
