import argparse
import numpy as np
from scipy.stats import norm
import sklearn.datasets
import sklearn.model_selection

import sys
sys.path.append(".")
from config import RANDOM_SEED

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", default=0.1, type=float, help="Smoothing parameter of our NB classifier")
parser.add_argument("--naive_bayes_type", default="bernoulli", choices=["gaussian", "multinomial", "bernoulli"])
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")

# In Gaussian native bayes, P(x| Ck) is modeled as a normal distribution. 
# The training phase consists of estimating the mean and variance (using MLE) to find the value of the normal distribution
def guassian(train_data, train_target, test_data, alpha, num_of_classes):
    N = test_data.shape[0]
    features_count = train_data.shape[1]
    means = np.zeros((num_of_classes,features_count))
    variances = np.zeros((num_of_classes,features_count))

    for k in range(num_of_classes):
        class_data = train_data[train_target == k]
        Nk = class_data.shape[0]
        # the estimated mean is the empirical mean
        means[k,:] = np.sum(class_data, axis=0) / Nk
        # the var is smoothed by a constant alpha to avoid sharp distributions
        variances[k,:] = np.sum((class_data - means[k,:])**2, axis=0) / Nk + alpha

    # now that we estimated the mean and the variance, we can plug it in the normal distribution 
    # formula to obtain the final estimation of P(X| Ck), and then use that to find the max arg of 
    # P(Ck | X) = P(Ck)P(X | Ck). Since multiplying many small probabilities can cause underflow, we take the log:
    # maxarg log( P(Ck | X) )= log(P(Ck)) + log(P(X | Ck))
    # the log log likelihood:
    #  sum(log(P(x_d | C_k)))

    targets_per_class_count = np.bincount(train_target)
    # The class prior is the distribution of the train data classes (assume uniform distribution)
    log_class_priors = np.log(targets_per_class_count / len(train_target))
    log_posteriors = np.zeros((N,num_of_classes))

    
    for k in range(num_of_classes):
        #log(P(X | Ck))
        log_norm = np.sum(norm.logpdf(test_data, loc=means[k], scale=np.sqrt(variances[k])), axis=1) 
        log_posteriors[:, k] = log_norm + log_class_priors[k]
    
    predictions = np.argmax(log_posteriors, axis=1)
    return predictions, log_posteriors


# In Bernoulli native bayes, P(x| Ck) is modeled using Bernoulli distribution:. 
# P(x| Ck) = Px,k ^ d * (1 - Px,k)^(1-b) where Px,k is the probability that input x is in class k and d is a binary value 1 or 0 for the target to take
# by setting the derivative of NLE to 0 we get the calculation below (with added smoothing)
def bernoulli(train_data, train_target, test_data, alpha, num_of_classes, threshold=8):
    N = test_data.shape[0]
    D = train_data.shape[1]
    probabilities = np.zeros((num_of_classes, D))
    targets_per_class_count = np.bincount(train_target)

    # Estimation of p_{d,k} with Laplace smoothing
    for k in range(num_of_classes):
        class_data = train_data[train_target == k]
        # Since Bernoulli NB works with binary data, binarize the features s.t 
        # a feature becomes 1 if the value at that location is smaller than some threshold and 0 otherwise
        binarized_class_data = (class_data >= threshold).astype(int)
        probabilities[k,:] = (np.sum(binarized_class_data,axis=0) + alpha) / (targets_per_class_count[k]+2*alpha)

    log_class_priors = np.log(targets_per_class_count / len(train_target))
    test_per_class_count = (test_data >= threshold).astype(int)
    log_posteriors = np.zeros((N,num_of_classes))
    
    for k in range(num_of_classes):
        #log(P(X | Ck))
        log_likelihood = np.sum(
            test_per_class_count * np.log(probabilities[k, :]) + (1 - test_per_class_count) * np.log(1 - probabilities[k, :]), 
            axis=1)
        log_posteriors[:, k] = log_likelihood + log_class_priors[k]
    
    predictions = np.argmax(log_posteriors, axis=1)
    return predictions, log_posteriors

def multinomial(train_data, train_target, test_data, alpha, num_of_classes):
    N = test_data.shape[0]
    D = train_data.shape[1]
    probabilities = np.zeros((num_of_classes, D))

    for k in range(num_of_classes):
        class_data = train_data[train_target == k]
        probabilities[k,:] = (np.sum(class_data,axis=0) + alpha) / (np.sum(class_data)+D*alpha)
        
    targets_per_class_count = np.bincount(train_target, minlength=num_of_classes)
    log_class_priors = np.log(targets_per_class_count / len(train_target))

    log_posteriors = np.zeros((N,num_of_classes))    
    
    for k in range(num_of_classes):
        #  log(P(C_k))
        # log likelihood: sum(test_data * log(P(x_d | C_k)))
        log_likelihood = np.sum(test_data * np.log(probabilities[k, :]), axis=1)
        log_posteriors[:, k] = log_likelihood + log_class_priors[k]
    
    predictions = np.argmax(log_posteriors, axis=1)
    return predictions, log_posteriors

def main(args: argparse.Namespace) -> tuple[float, float]:
    # Load the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=RANDOM_SEED)
    
    # Need to solve bayes model:
    # max_arg Ck of P(Ck | X) = P(Ck)P(X | Ck) which can be expanded to (mult for every input x in X)P(Ck)P(x| Ck)

    predictions = []
    log_posteriors = []
    bayes_type = args.naive_bayes_type
    
    if (bayes_type == "gaussian"):
        predictions, log_posteriors = guassian(train_data, train_target, test_data, args.alpha, args.classes)

    elif (bayes_type == "multinomial"):
        predictions, log_posteriors = multinomial(train_data, train_target, test_data, args.alpha, args.classes)
    else:
        predictions, log_posteriors = bernoulli(train_data, train_target, test_data, args.alpha, args.classes)


    correct_predictions = np.sum(predictions == test_target)
    test_accuracy = correct_predictions / len(test_target)
    # Compute joint log-probability
    test_log_probability = np.sum(log_posteriors[np.arange(len(test_target)), test_target])

    return 100 * test_accuracy, test_log_probability

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy, test_log_probability = main(main_args)

    print("Test accuracy {:.2f}%, log probability {:.2f}".format(test_accuracy, test_log_probability))