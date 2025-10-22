import argparse
import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import matplotlib.pyplot as plt

import sys
sys.path.append(".")
from config import RANDOM_SEED

parser = argparse.ArgumentParser()
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")

def main(args: argparse.Namespace) -> tuple[float, float]:
    dataset = sklearn.datasets.load_diabetes()
    X = dataset.data
    t = dataset.target
    X_train, X_test, t_train, t_test = sklearn.model_selection.train_test_split(X, t, test_size=args.test_size, random_state=RANDOM_SEED)
    # Geomspace generates an array of 500 values between 0.01 and 10. Geometrically spaced numbers increase (or decrease) by a constant multiplier, rather than by addition. 
    # This is equivalent to spacing on a logarithmic scale, e.g., 1, 10, 100, 1000, 10000.
    lambdas = np.geomspace(0.01, 10, num=500)
    # Ridge regression is a type of linear regression that includes L2 regularization to prevent overfitting and improve model generalization.
    best_lambda = None
    best_rmse = float('inf')
    rmses = []
    # For every alpha, I compute the root mean squared error and return the lambda producing lowest one and the corresponding value
    for l in lambdas:
        model = sklearn.linear_model.Ridge(alpha= l)
        #train
        model.fit(X_train,t_train)
        #predict
        predictions = model.predict(X_test)
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(predictions, t_test))
        rmses.append(rmse)
        if(rmse < best_rmse):
            best_rmse = rmse
            best_lambda = l


    if args.plot:
        plt.plot(lambdas, rmses)
        plt.xscale("log")
        plt.xlabel("L2 regularization strength $\\lambda$")
        plt.ylabel("RMSE")
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return best_lambda, best_rmse


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    best_lambda, best_rmse = main(main_args)
    print("{:.2f} {:.2f}".format(best_lambda, best_rmse))
