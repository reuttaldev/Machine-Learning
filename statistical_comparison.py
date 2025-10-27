import argparse
import warnings
import numpy as np
import sklearn.datasets
import sklearn.exceptions
import sklearn.linear_model
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sys
sys.path.append(".")
from config import RANDOM_SEED

parser = argparse.ArgumentParser()
parser.add_argument("--method", default="bootstrap", choices=["bootstrap", "permutation"],
                    help="Statistical test method to run")
parser.add_argument("--bootstrap_samples", default=1000, type=int, help="Bootstrap resamplings")
parser.add_argument("--random_samples", default=1000, type=int, help="Number of random permutations")
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--test_size", default=0.5,
                    type=lambda x: int(x) if x.isdigit() else float(x),
                    help="Test size")

def main(args: argparse.Namespace):
    warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)
    generator = np.random.RandomState(RANDOM_SEED)

    # Load dataset
    data = sklearn.datasets.load_digits(n_class=args.classes)
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data.data, data.target, test_size=args.test_size, random_state=RANDOM_SEED
    )

    # Train two Logistic Regression models with different polynomial feature degrees
    models, predictions = [], []
    for degree in [1, 2]:
        model = sklearn.pipeline.Pipeline([
            ("features", sklearn.preprocessing.PolynomialFeatures(degree=degree)),
            ("estimator", sklearn.linear_model.LogisticRegression(solver="saga", random_state=RANDOM_SEED)),
        ])
        model.fit(train_data, train_target)
        models.append(model)
        predictions.append(model.predict(test_data))

    # -------------------------------------------------------------
    # Bootstrap Resampling Test
    # -------------------------------------------------------------
    def bootstrap_resampling():
        bootstrap_scores = np.zeros((2, args.bootstrap_samples))
        diff_count = 0

        for i in range(args.bootstrap_samples):
            indices = generator.choice(len(test_data), size=len(test_data), replace=True)
            for m in range(2):
                preds = predictions[m][indices]
                bootstrap_scores[m, i] = np.mean(preds == test_target[indices]) * 100

            # count how often model2 fails to outperform model1
            if bootstrap_scores[1, i] - bootstrap_scores[0, i] <= 0:
                diff_count += 1

        confidence = 2.5
        confidence_intervals = [
            (np.percentile(bootstrap_scores[0], confidence),
             np.percentile(bootstrap_scores[0], 100 - confidence)),
            (np.percentile(bootstrap_scores[1], confidence),
             np.percentile(bootstrap_scores[1], 100 - confidence)),
        ]
        result = diff_count / args.bootstrap_samples * 100
        return confidence_intervals, result

    # -------------------------------------------------------------
    # Permutation Test
    # -------------------------------------------------------------
    def permutation_test():
        # Predictions from both models on the same test data
        preds = np.vstack((models[0].predict(test_data), models[1].predict(test_data)))
        scores = np.zeros(args.random_samples)

        for i in range(args.random_samples):
            # Randomly assign which model’s prediction to take for each sample
            assignments = generator.choice(2, size=len(test_data), replace=True)
            mixed_preds = preds[assignments, np.arange(len(test_data))]
            scores[i] = np.mean(test_target == mixed_preds) * 100

        second_model_score = np.mean(test_target == preds[1, :]) * 100
        # One-sided p-value: how often a random mix matches/exceeds model2’s accuracy
        p_value = np.mean(scores >= second_model_score) * 100
        return p_value

    # Run test based on argument
    if args.method == "bootstrap":
        confidence_intervals, bootstrap_result = bootstrap_resampling()
        print("Confidence intervals of the two models:")
        for interval in confidence_intervals:
            print("- [{:.2f}% .. {:.2f}%]".format(*interval))
        print(f"Null hypothesis probability (bootstrap): {bootstrap_result:.2f}%")
    else:
        p_value = permutation_test()
        print(f"Permutation test p-value: {p_value:.2f}%")

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
