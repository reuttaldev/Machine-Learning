
import argparse

import numpy as np
import sklearn.datasets
import sklearn.model_selection

import sys
sys.path.append(".")
from config import RANDOM_SEED

parser = argparse.ArgumentParser()
parser.add_argument("--criterion", default="gini", type=str, help="Criterion to use; either `gini` or `entropy`")
parser.add_argument("--dataset", default="wine", type=str, help="Dataset to use")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--max_leaves", default=None, type=int, help="Maximum number of leaf nodes")
parser.add_argument("--min_to_split", default=2, type=int, help="Minimum examples required to split")
parser.add_argument("--test_size", default=0.25, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")

def most_frequent_class(labels):
    # - For each node, predict the most frequent class (and the one with
    #   the smallest number if there are several such classes).
    values, counts = np.unique(labels, return_counts=True)
    max_count = counts.max()
    max_classes = values[counts == max_count]
    # returning the smallest class among those with the maximum count
    return max_classes.min()

def compute_criterion(target, criterion):
    _, counts = np.unique(target, return_counts=True)
    probabilities = counts / len(target)
    if criterion == "gini":
        return len(target) * (1 - np.sum(probabilities ** 2))
    elif criterion == "entropy":
        return len(target) * -np.sum(probabilities * np.log2(probabilities + 1e-10))

def split_data(data, target, feature_index, split_value):
    # - When splitting a node, consider the features in sequential order, then
    #   for each feature consider all possible split points ordered in ascending
    #   value, and perform the first encountered split decreasing the criterion
    #   the most. Each split point is an average of two nearest unique feature values
    #   of the instances corresponding to the given node (e.g., for four instances
    #   with values 1, 7, 3, 3, the split points are 2 and 5).

    # two boolean arrays indicating wether the index should go to the left or right subtree, 
    # splitting based on the feature (using its index) and the split value
    left_node_mask = data[:, feature_index] < split_value
    right_node_mask = ~left_node_mask
    return (data[left_node_mask], target[left_node_mask]), (data[right_node_mask], target[right_node_mask])

def split_node(data, target, criterion):
    best_split = None
    best_criterion_decrease = -np.inf
    current_criterion_value = compute_criterion(target, criterion)

    for feature_index in range(data.shape[1]):
        unique_values = np.unique(data[:, feature_index])
        if len(unique_values) > 1:
            split_points = (unique_values[:-1] + unique_values[1:]) / 2
        else:
            continue

        for split_value in split_points:
            (left_data, left_target), (right_data, right_target) = split_data(data, target, feature_index, split_value)

            # skiping any empty
            if len(left_target) != 0 and len(right_target) != 0:
                left_criterion = compute_criterion(left_target, criterion)
                right_criterion = compute_criterion(right_target, criterion)
                weighted_criterion = left_criterion + right_criterion

                criterion_decrease = current_criterion_value - weighted_criterion

                # update if this split is the best so far
                if criterion_decrease > best_criterion_decrease:
                    best_split = (feature_index, split_value, left_data, left_target, right_data, right_target)
                    best_criterion_decrease = criterion_decrease
    
    return best_split, best_criterion_decrease

def make_node_non_recursive(data, target, max_depth, max_leaves, min_to_split, criterion):
    # non-recursive tree construction

    current_leaf_nodes = []

    root = {
        "is_leaf": True,
        "data": data,
        "target": target,
        "depth": 0,
        "prediction": most_frequent_class(target),
    }
    current_leaf_nodes.append(root)

    # - Allow splitting a node only if:
    #   - when `args.max_depth` is not `None`, its depth must be less than `args.max_depth`
    #     (depth of the root node is zero);
    #   - when `args.max_leaves` is not `None`, there are less than `args.max_leaves` leaves
    #     (a leaf is a tree node without children);
    #   - there are at least `args.min_to_split` corresponding instances;
    #   - the criterion value is not zero.
    while True:
        # if the number of leaves meets the max_leaves constraint, stop splitting
        if len(current_leaf_nodes) >= max_leaves:
            break

        # searching the best leaf node to split
        best_split_node = None
        best_split = None
        best_criterion_decrease = -np.inf

        for node in current_leaf_nodes:
            # skip non-leaf nodes or nodes that cannot be split further (constraints aren't valid)
            if ((max_depth is not None and node["depth"] >= max_depth) or 
                len(node["target"]) < min_to_split or 
                compute_criterion(node["target"], criterion) == 0
            ):
                continue

            split, criterion_decrease = split_node(node["data"], node["target"], criterion)

            # update if this split is the best so far
            if split and criterion_decrease > best_criterion_decrease:
                best_split_node = node
                best_split = split
                best_criterion_decrease = criterion_decrease

        # no valid split was found => stop splitting
        if best_split_node is None:
            break

        feature_index, split_value, left_data, left_target, right_data, right_target = best_split

        left_child = {
            "is_leaf": True,
            "data": left_data,
            "target": left_target,
            "depth": best_split_node["depth"] + 1,
            "prediction": most_frequent_class(left_target),
        }

        right_child = {
            "is_leaf": True,
            "data": right_data,
            "target": right_target,
            "depth": best_split_node["depth"] + 1,
            "prediction": most_frequent_class(right_target),
        }

        # update and replace the split node with its children in the list of current leaves
        best_split_node.update({
            "is_leaf": False,
            "feature_index": feature_index,
            "split_value": split_value,
            "left": left_child,
            "right": right_child,
        })

        current_leaf_nodes.remove(best_split_node)
        current_leaf_nodes.extend([left_child, right_child])

    # remove data and target attributes from all nodes for a cleaner tree (may be skipped)
    def clean_tree(node):
        if "data" in node:
            del node["data"]
        if "target" in node:
            del node["target"]
        if not node["is_leaf"]:
            clean_tree(node["left"])
            clean_tree(node["right"])

    clean_tree(root)
    return root

def make_node_recursive(data, target, depth, max_depth, min_to_split, criterion):
    # recursive tree construction for unlimited leaves

    # - Allow splitting a node only if:
    #   - when `args.max_depth` is not `None`, its depth must be less than `args.max_depth`
    #     (depth of the root node is zero);
    #   - when `args.max_leaves` is not `None`, there are less than `args.max_leaves` leaves
    #     (a leaf is a tree node without children);
    #   - there are at least `args.min_to_split` corresponding instances;
    #   - the criterion value is not zero.
    if (
        (max_depth is not None and depth >= max_depth)
        or len(target) < min_to_split
        or compute_criterion(target, criterion) == 0
    ):
        return {"is_leaf": True, "prediction": most_frequent_class(target)}

    best_split, _ = split_node(data, target, criterion)
    if best_split is None:
        return {"is_leaf": True, "prediction": most_frequent_class(target)}

    feature_index, split_value, left_data, left_target, right_data, right_target = best_split
    return {
        "is_leaf": False,
        "feature_index": feature_index,
        "split_value": split_value,
        "left": make_node_recursive(left_data, left_target, depth + 1, max_depth, min_to_split, criterion),
        "right": make_node_recursive(right_data, right_target, depth + 1, max_depth, min_to_split, criterion),
    }


def decision_tree(train_data, train_target, args):
    # - When `args.max_leaves` is `None`, use recursive (left descendants first, then
    #   right descendants) approach, splitting every node if the constraints are valid.
    #   Otherwise (when `args.max_leaves` is not `None`), repeatably split a leaf where the
    #   constraints are valid and the overall criterion value ($c_left + c_right - c_node$)
    #   decreases the most. If there are several such nodes, choose the one
    #   which was created sooner (a left child is considered to be created
    #   before a right child).

    if args.max_leaves is None:       
        return make_node_recursive(
            data=train_data, 
            target=train_target, 
            depth=0, 
            max_depth=args.max_depth,
            min_to_split=args.min_to_split,
            criterion=args.criterion,)
    
    # otherwise
    return make_node_non_recursive(
        data=train_data,
        target=train_target,
        max_depth=args.max_depth,
        max_leaves=args.max_leaves,
        min_to_split=args.min_to_split,
        criterion=args.criterion
    )

def main(args: argparse.Namespace) -> tuple[float, float]:
    def predict(tree, sample):
        if tree["is_leaf"]:
            return tree["prediction"]
        if sample[tree["feature_index"]] < tree["split_value"]:
            return predict(tree["left"], sample)
        else:
            return predict(tree["right"], sample)
    
    # Use the given dataset.
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # TODO: Manually create a decision tree on the training data.
    
    tree = decision_tree(
        train_data=train_data,
        train_target=train_target,
        args=args
    )

    train_predictions = np.array([predict(tree, sample) for sample in train_data])
    train_accuracy = np.mean(train_predictions == train_target)

    test_predictions = np.array([predict(tree, sample) for sample in test_data])
    test_accuracy = np.mean(test_predictions == test_target)

    return 100 * train_accuracy, 100 * test_accuracy


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(main_args)

    print("Train accuracy: {:.1f}%".format(train_accuracy))
    print("Test accuracy: {:.1f}%".format(test_accuracy))