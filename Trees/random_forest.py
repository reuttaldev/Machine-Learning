import argparse
import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import scipy

import sys
sys.path.append(".")
from config import RANDOM_SEED

parser = argparse.ArgumentParser()
parser.add_argument("--bagging", default=False, action="store_true", help="Perform bagging")
parser.add_argument("--dataset", default="wine", type=str, help="Dataset to use")
parser.add_argument("--feature_subsampling", default=1.0, type=float, help="What fraction of features to subsample")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--test_size", default=0.25, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
parser.add_argument("--trees", default=1, type=int, help="Number of trees in the forest")

class Node:
    def __init__(self, is_leaf, data, target, prediction=None, left=None, right=None, feature_index=None, split_value=None):
        self.is_leaf = is_leaf
        self.data = data
        self.target = target
        self.prediction = prediction
        self.left = left
        self.right = right
        self.feature_index = feature_index
        self.split_value = split_value


def most_frequent_class(labels):
    values, counts = np.unique(labels, return_counts=True)
    max_count = counts.max()
    max_classes = values[counts == max_count]
    return max_classes.min()

def compute_criterion(target):
    _, counts = np.unique(target, return_counts=True)
    probabilities = counts / len(target)
    return -len(target) * np.sum(probabilities * np.log2(probabilities))

# Splitting the dataset based on a feature threshold
def split_data(data, target, feature_index, split_value):

    left_node_mask = data[:, feature_index] < split_value
    right_node_mask = ~left_node_mask
    return (data[left_node_mask], target[left_node_mask]), (data[right_node_mask], target[right_node_mask])

# Searching for the best feature and threshold that minimizes the criterion
def split_node(data, target, feature_indices):
    best_split = None
    best_criterion_decrease = -np.inf
    current_criterion_value = compute_criterion(target)

    for feature_index in feature_indices:
        unique_values = np.unique(data[:, feature_index])
        if len(unique_values) > 1:
            split_points = (unique_values[:-1] + unique_values[1:]) / 2
        else:
            continue

        for split_value in split_points:
            (left_data, left_target), (right_data, right_target) = split_data(data, target, feature_index, split_value)

            # skip any empty splits
            if len(left_target) != 0 and len(right_target) != 0:
                left_criterion = compute_criterion(left_target)
                right_criterion = compute_criterion(right_target)
                weighted_criterion = left_criterion + right_criterion

                criterion_decrease = current_criterion_value - weighted_criterion

                # update if this split is the best so far
                if criterion_decrease > best_criterion_decrease:
                    best_split = (feature_index, split_value, left_data, left_target, right_data, right_target)
                    best_criterion_decrease = criterion_decrease

    return best_split

# Build a decision tree recursively 
def make_node_recursive(data, target, depth, max_depth, subsample_features):
  
    if depth >= max_depth or compute_criterion(target) <= 1e-10:
        return Node(
            is_leaf=True,
            data=data,
            target=target,
            prediction=most_frequent_class(target),
        )
    
    feature_indices = subsample_features(data.shape[1])
    
    best_split = split_node(data, target, feature_indices)
    if best_split is None:
        return Node(
            is_leaf=True,
            data=data,
            target=target,
            prediction=most_frequent_class(target),
        )
    
    feature_index, split_value, left_data, left_target, right_data, right_target = best_split

    left_child = make_node_recursive(
        data=left_data,
        target=left_target,
        depth=depth + 1,
        max_depth=max_depth,
        subsample_features=subsample_features
    )

    right_child = make_node_recursive(
        data=right_data,
        target=right_target,
        depth=depth + 1,
        max_depth=max_depth,
        subsample_features=subsample_features
    )

    return Node(
        is_leaf=False,
        data=data,
        target=target,
        left=left_child,
        right=right_child,
        feature_index=feature_index,
        split_value=split_value,
    )


def decision_tree(train_data, train_target, args, subsample_features):
    return make_node_recursive(
        data=train_data,
        target=train_target,
        depth=0,
        max_depth=args.max_depth,
        subsample_features=subsample_features
    )


def main(args: argparse.Namespace) -> tuple[float, float]:
    def predict(tree, sample):
        if tree.is_leaf:
            return tree.prediction
        if sample[tree.feature_index] < tree.split_value:
            return predict(tree.left, sample)
        else:
            return predict(tree.right, sample)
        
    def predict_dataset(forest, data):
        predictions = np.array([[predict(tree, sample) for tree in forest] for sample in data])
        return scipy.stats.mode(predictions, axis=1, keepdims=True).mode.flatten()

    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=RANDOM_SEED)
    
    generator_feature_subsampling = np.random.RandomState(RANDOM_SEED)
    def subsample_features(number_of_features: int) -> np.ndarray:
        return np.sort(generator_feature_subsampling.choice(
            number_of_features, size=int(args.feature_subsampling * number_of_features), replace=False))

    generator_bootstrapping = np.random.RandomState(RANDOM_SEED)
    def bootstrap_dataset(train_data: np.ndarray) -> np.ndarray:
        return generator_bootstrapping.choice(len(train_data), size=len(train_data), replace=True)
    
    forest = []
    for _ in range(args.trees):
        if args.bagging:
            indices = bootstrap_dataset(train_data)
            subset_train_data = train_data[indices]
            subset_train_target = train_target[indices]
        else:
            subset_train_data = train_data
            subset_train_target = train_target

        tree = decision_tree(
            train_data=subset_train_data,
            train_target=subset_train_target,
            args=args,
            subsample_features=subsample_features
        )
        forest.append(tree)

    # Measure the training and testing accuracy.
    train_predictions = predict_dataset(forest, train_data)
    test_predictions = predict_dataset(forest, test_data)
    
    train_accuracy = sklearn.metrics.accuracy_score(train_target, train_predictions)
    test_accuracy = sklearn.metrics.accuracy_score(test_target, test_predictions)

    return 100 * train_accuracy, 100 * test_accuracy

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(main_args)

    print("Train accuracy: {:.1f}%".format(train_accuracy))
    print("Test accuracy: {:.1f}%".format(test_accuracy))