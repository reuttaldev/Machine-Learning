import argparse
import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--epochs", default=50, type=int, help="Number of SGD training epochs")
parser.add_argument("--l2", default=0.0, type=float, help="L2 regularization strength")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
RANDOM_SEED = 42

def main(args: argparse.Namespace) -> tuple[list[float], float, float]:

    # Generating an artificial regression dataset.
    X, t = sklearn.datasets.make_regression(n_samples=args.data_size, random_state=RANDOM_SEED)

    # Addding constant feature with value 1 to represent the bias
    X = np.concatenate((X, np.ones(shape= (X.shape[0],1))),axis = 1)
    X_train, X_test, t_train, t_test = sklearn.model_selection.train_test_split(X, t, test_size=args.test_size, random_state=RANDOM_SEED)
    generator = np.random.RandomState(RANDOM_SEED)
    # Generating initial linear regression weights. We need the same amount of weights as we have features in our data, i.e. the number of columns in X_train
    weights = generator.uniform(size=X_train.shape[1], low=-0.1, high=0.1)

    train_rmses, test_rmses = [], []
    # epoch in machine learning refers to one complete pass through the entire training dataset 
    for epoch in range(args.epochs):
        permutation = generator.permutation(X_train.shape[0])
        # Processing the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.         
        for batch_count in range(0, len(permutation), args.batch_size):
            # Choosing fitst args.batch_size from premutation, starting from the step we are in now
            batch_indices = permutation[batch_count : batch_count+args.batch_size]
            # Getting the batch based on the indicies.
            X_batch = X_train[batch_indices]
            t_batch = t_train[batch_indices]

            # Finding the gradients for this batch. the gradient is the direction of steeperst asend with respect to each of the weights
            batch_gradient = np.zeros(weights.shape)
            for x_i,t_i in zip(X_batch, t_batch):   
                # The gradient for the input example $(x_i, t_i)$ is
                # - $(x_i^T weights - t_i) x_i$ for the unregularized loss (1/2 MSE loss),
                # - $args.l2 * weights_with_bias_set_to_zero$ for the L2 regularization argument,
                #   where we set the bias to zero because the bias should not be regularized

                # x_1 is a feature vector of size 1xn. weights is the n*1 vector we are trying to optimize. t is the expected (target) value for x_1^T*w. Since it is a multiplicatoin of 2 vectors of equal dimention we do dot product and get a single numerical value. 
                prediction = np.dot(x_i,weights)#x_i.T will yield the same result as x_i itself since it is a 1d array
                error =  prediction - t_i 
                loss_gradient = x_i * error
                weights_with_bias_set_to_zero = np.hstack([weights[:-1],0])
                regularzation = args.l2 * weights_with_bias_set_to_zero
                batch_gradient += loss_gradient+ regularzation

            batch_gradient /= args.batch_size
            # Updating the weights accordinaly -- the SGD update is weights = weights - args.learning_rate * gradient
            weights -= args.learning_rate * batch_gradient

        # Appendding current RMSE on train/test to `train_rmses`/`test_rmses`.
        train_predictions = X_train @ weights 
        test_predictions = X_test @ weights
        train_rmses.append(sklearn.metrics.root_mean_squared_error(train_predictions,t_train))
        test_rmses.append(sklearn.metrics.root_mean_squared_error(test_predictions,t_test))

    # Computing explicit test data RMSE when fitting for comparison
    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, t_train)
    explicit_predictions = model.predict(X_test)
    explicit_rmse = sklearn.metrics.root_mean_squared_error(explicit_predictions, t_test)

    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(train_rmses, label="Train")
        plt.plot(test_rmses, label="Test")
        plt.xlabel("Epochs")
        plt.ylabel("RMSE")
        plt.legend()
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return weights, test_rmses[-1], explicit_rmse


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, sgd_rmse, explicit_rmse = main(main_args)
    print("Test RMSE: SGD {:.3f}, explicit {:.1f}".format(sgd_rmse, explicit_rmse))
    print("Learned weights:", *("{:.3f}".format(weight) for weight in weights[:12]), "...")
