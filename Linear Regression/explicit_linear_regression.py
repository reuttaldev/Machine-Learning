import argparse
import numpy as np
import sklearn.datasets
import sklearn.model_selection

parser = argparse.ArgumentParser()
parser.add_argument("--test_size", default=0.1, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")

RANDOM_SEED = 42
def main(args: argparse.Namespace) -> float:
    dataset = sklearn.datasets.load_diabetes()
    X = dataset.data
    t = dataset.target
    # Append a feature with value 1 to the end of every input data. Then the bias becoms the last column of the y(x) matrix.
    num_of_rows = X.shape[0]
    ones_column = np.ones((num_of_rows,1))
    X = np.concatenate((X, ones_column), axis = 1) 
    X_train, X_test, t_train, t_test = sklearn.model_selection.train_test_split(X, t, test_size=args.test_size, random_state=RANDOM_SEED)
    # Predict target values on the test set.
    weights = linear_regression_weights(X_train,t_train)
    function_coefficient = X_test @ weights
    # TODO: Manually compute root mean square error on the test set predictions.
    rmse = RMSE(function_coefficient, t_test)
    return rmse

# The goal of linear regression is to find the function y = M*X + b which best fits (minimizes MSE) our input points. 
# Here, we find the exact solution of the minimum value of the MSE loss function, by finding where the gradient is equal to 0 
def linear_regression_weights(X,t):
    return np.linalg.inv(X.T @ X) @ X.T @ t 

def RMSE(prediction, target):
    squr_diff  = (target - prediction) ** 2  
    mean = np.mean(squr_diff)
    return np.sqrt(mean)
    
if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    rmse = main(args)
    print("{:.2f}".format(rmse))
