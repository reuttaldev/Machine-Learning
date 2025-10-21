import argparse
import sklearn.datasets
import sklearn.linear_model
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
from config import RANDOM_SEED
import warnings
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")


def main(args: argparse.Namespace) -> float:
    # Loading the digits dataset
    dataset = sklearn.datasets.load_digits()
    dataset.target = dataset.target % 2
    X = dataset.data
    t = dataset.target
    X_train, X_test, t_train, t_test = sklearn.model_selection.train_test_split(X,t,test_size=args.test_size, random_state=RANDOM_SEED)


    pipeline = sklearn.pipeline.Pipeline([
        ("scaler", sklearn.preprocessing.MinMaxScaler()),
        ("poly_features", sklearn.preprocessing.PolynomialFeatures()),
        ("log_reg", sklearn.linear_model.LogisticRegression(random_state=RANDOM_SEED         
        ))
    ])
    # Evaluate crossvalidated train performance of all combinations of the following parameters:
    # - polynomial degree: 1, 2
    # - LogisticRegression regularization C: 0.01, 1, 100
    # - LogisticRegression solver: lbfgs, sag
    param_grid = {
        "poly_features__degree": [1, 2],
        "log_reg__C": [0.01, 1, 100],
        "log_reg__solver": ["lbfgs", "sag"]
    }
    grid_search = sklearn.model_selection.GridSearchCV(
        pipeline,
        param_grid,
        cv=sklearn.model_selection.StratifiedKFold(n_splits=5),
        scoring="accuracy",
        return_train_score=True
    )
    grid_search.fit(X_train, t_train)
    # For the best combination of parameters, computing the test set accuracy.
    best_model = grid_search.best_estimator_
    # The score method internally uses the model to make predictions on the test data (X_test) and compares those predictions to the true labels (t_test).
    test_accuracy = best_model.score(X_test, t_test)

    # Showing the results of all the hyperparameter values evaluated:
    for rank, accuracy, params in zip(grid_search.cv_results_["rank_test_score"],
                                        grid_search.cv_results_["mean_test_score"],
                                        grid_search.cv_results_["params"]):
        print("Rank: {:2d} Cross-val: {:.1f}%".format(rank, 100 * accuracy),
                *("{}: {:<5}".format(key, value) for key, value in params.items()))

    return 100 * test_accuracy


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy = main(main_args)
    print("Test accuracy: {:.2f}%".format(test_accuracy))
