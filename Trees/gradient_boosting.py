import argparse
import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sys
sys.path.append(".")
from config import RANDOM_SEED

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="wine", type=str, help="Dataset to use")
parser.add_argument("--l2", default=1., type=float, help="L2 regularization factor")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate")
parser.add_argument("--max_depth", default=1, type=int, help="Maximum decision tree depth")
parser.add_argument("--test_size", default=0.25, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
parser.add_argument("--trees", default=1, type=int, help="Number of trees in the forest")

class Node:
    def __init__(self, data_indices,depth):
        self.is_leaf = True
        self.left = None
        self.right = None
        # indices of data samples that belongs to this Node
        self.data_indices = data_indices
        self.split_threshold  = None
        self.split_feature_index = None
        self.criterion = 0.0 
        self.prediction = 0.0 
        self.depth = depth
    
    # Searching for the best feature and threshold that minimizes the criterion
    def find_best_split(self,data, g,h,l2):

        child_indices = None
        best_criterion_decrease = np.inf
        split_feature_index = -1
        split_threshold = -1

        # Trying all features & midpoints
        for c in range(data.shape[1]):
            # features in data that belongs to this node
            features_in_col = data[self.data_indices, c]
            unique_values = np.unique(features_in_col)
            if len(unique_values) > 1:
                # splitting based on value 
                split_points = (unique_values[:-1] + unique_values[1:]) / 2
            else:
                continue
            for split in split_points:
                left_indices = features_in_col <= split
                left_g = g[left_indices]
                right_g = g[~left_indices]
                # skip invalid (empty) splits
                if len(left_g) == 0 or len(right_g) == 0:
                    continue
                left_h = h[left_indices]
                right_h = h[~left_indices]

                left_criterion = calculate_criterion(left_g,left_h,l2)
                right_criterion = calculate_criterion(right_g,right_h,l2)
                criterion_decrease = left_criterion + right_criterion - self.criterion

                # update if this split is the best so far
                if criterion_decrease < best_criterion_decrease:
                    best_criterion_decrease = criterion_decrease
                    child_indices = (self.data_indices[left_indices],self.data_indices[~left_indices])
                    split_feature_index = c
                    split_threshold = split

        if child_indices is None:
            return None
        return child_indices, split_feature_index,split_threshold

    def split(self, child_indices,feature, threshold):
        self.is_leaf = False
        self.left = Node(child_indices[0],self.depth +1)
        self.right = Node(child_indices[1],self.depth +1)
        self.split_threshold = threshold
        self.split_feature_index = feature
    
    def predict(self, x):
        if self.is_leaf:
            return self.prediction
        else:
            if x[self.split_feature_index] <= self.split_threshold:
                return self.left.predict(x)
            else:
                return self.right.predict(x)      

class Tree:
    def __init__(self,max_depth,l2):
        self.max_depth= max_depth
        self.min_to_split = 2
        self.l2 = l2
        self.root = None

    def leaf_optimal_weight(self,residuals,second_deriv_residuals):
        return -np.sum(residuals) / (self.l2 + np.sum(second_deriv_residuals))
    
    def predict(self, X):
        return np.array([self.root.predict(x) for x in X], dtype=np.float64)

    # Construct and train a tree
    def fit(self,X,residuals,second_deriv_residuals):
        N = X.shape[0]
        #Init root with all indices
        self.root = Node(np.arange(N),0)
        stack = [self.root]

        while stack:
            node = stack.pop()
            # if not enough data to split
            if self.min_to_split > len(node.data_indices):
                continue
            #if too deep, or  there is more than 1 example corresponding to it --
            #  a non-zero criterion value in the previous assignments)
            if node.depth >= self.max_depth:
                continue
            # filter rows
            node_residuals = residuals[node.data_indices]
            node_second_d_residuals = second_deriv_residuals[node.data_indices]
            node.criterion = calculate_criterion(node_residuals, node_second_d_residuals,self.l2)
            if node.criterion == 0:
                continue
            node.prediction = self.leaf_optimal_weight(node_residuals, node_second_d_residuals)
            # search for best split
            child_indices, feature, threshold = node.find_best_split(X,node_residuals,node_second_d_residuals,self.l2)
            if(child_indices is None):
                continue
            # perform it
            node.split(child_indices, feature, threshold)
            left_idx, right_idx = child_indices
            node.left.prediction  = self.leaf_optimal_weight(residuals[left_idx], second_deriv_residuals[left_idx])
            node.right.prediction = self.leaf_optimal_weight(residuals[right_idx], second_deriv_residuals[right_idx])
            stack.append(node.left)
            stack.append(node.right)
        
# this function is obtained from taking the loss function L(F) (e.g. logistic loss)
# and approximate it using a second-order Taylor expansion around current F, and plugging in the optimal weight value
def calculate_criterion(residuals,second_deriv_residuals,l2):
    return -0.5* (np.sum(residuals)**2 / (l2 + np.sum(second_deriv_residuals)))
    
def softmax(z):
    z -= np.max(z, axis=1, keepdims=True)
    z_exp = np.exp(z)
    return z_exp / np.sum(z_exp, axis=1, keepdims=True)

def main(args: argparse.Namespace) -> tuple[list[float], list[float]]:
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=RANDOM_SEED)

    K = len(np.unique(train_target))
    train_prediction = np.zeros((train_data.shape[0],K))
    test_prediction = np.zeros((test_data.shape[0],K))
    train_target_hot_coded = np.eye(K)[train_target]
    train_accuracies, test_accuracies = [], []

    for _ in range(args.trees):
        # need the predictions to add up to 1 to become probs
        probabilities = softmax(train_prediction)
        # for multiclass classification, we need to model the full categorical output distribution. 
        # So at each step train (K) regression trees, each predicting a single value of the linear part of the generalized linear model
        for k in range(K):
            probabilities_per_class = probabilities[:,k]
            #first derivative of NLL(onehot_target_i_c, softmax(y_{t-1}(x_i))_c) with respect to y_{t-1}(x_i)_c.
            residuals =  probabilities_per_class-train_target_hot_coded[:,k]
            second_deriv_residuals = probabilities_per_class*(1-probabilities_per_class)
            # create a new tree that will be train on the errors (some function of residuals and their derivative) of the predictions of the trees before them
            tree = Tree(args.max_depth, args.l2)
            tree.fit(train_data, residuals, second_deriv_residuals)
            # - compute the current predictions `y_{t-1}(x_i)` for every training example `i` as
            #   y_{t-1}(x_i)_c = \sum_{j=1}^{t-1} args.learning_rate * tree_{iter=j,class=c}.predict(x_i)
            train_prediction[:,k]  = train_prediction[:,k] + args.learning_rate * tree.predict(train_data)
            test_prediction[:,k]  = test_prediction[:,k] + args.learning_rate * tree.predict(test_data)

        train_accuracies.append(sklearn.metrics.accuracy_score(train_target, np.argmax(train_prediction, axis=1)))
        test_accuracies.append(sklearn.metrics.accuracy_score(test_target, np.argmax(test_prediction, axis=1)))

        # To perform a prediction using t trees, compute the y_t(x_i) and return the
        # class with the highest value (pick the smallest class number if there is a tie).
    
    return [100 * acc for acc in train_accuracies], [100 * acc for acc in test_accuracies]


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracies, test_accuracies = main(main_args)

    for i, (train_accuracy, test_accuracy) in enumerate(zip(train_accuracies, test_accuracies)):
        print("Using {} trees, train accuracy: {:.1f}%, test accuracy: {:.1f}%".format(
            i + 1, train_accuracy, test_accuracy))
