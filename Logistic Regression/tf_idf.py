import argparse
import lzma
import pickle
import os
import sys
import urllib.request
import sys
sys.path.append(".")
from config import RANDOM_SEED
import numpy as np
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import collections 
import re
parser = argparse.ArgumentParser()

parser.add_argument("--idf", default=False, action="store_true", help="Use IDF weights")
parser.add_argument("--tf", default=False, action="store_true", help="Use TF weights")
parser.add_argument("--test_size", default=500, type=int, help="Test set size")
parser.add_argument("--train_size", default=1000, type=int, help="Train set size")


class NewsGroups:
    def __init__(self,
                 name="20newsgroups.train.pickle",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        with lzma.open(name, "rb") as dataset_file:
            dataset = pickle.load(dataset_file)

        self.DESCR = dataset.DESCR
        self.data = dataset.data[:data_size]
        self.target = dataset.target[:data_size]
        self.target_names = dataset.target_names


def get_tf_idf(args,X,features, idf):
    # For each document, I compute (TF), if `args.tf` is set (term frequency is
    #   proportional to the number of term occurrences but normalized to
    #   sum to 1 over all features of a document);
    # - otherwise, use binary indicators (1 if a given term is present, else 0)
    all_doc_features = np.zeros((len(X),len(features)))
    # Using a dict to give O(1) acess to check if a term is a feature, otherwise the code is too slow
    features_dict = {feature: i for i, feature in enumerate(features)}
    for i,doc in enumerate(X):
        terms = re.findall(r'\w+', doc)
        counter_for_doc = collections.Counter(terms)
        filtered_terms = {term: count for term, count in counter_for_doc.items() if term in features_dict}
        # It is not the number of terms, or the occurances of the number of terms, but the number of occurances of features in this doc.
        # I figured it out bc it said to "normalized to sum to 1 over all features of a document)"
        num_of_terms = sum(filtered_terms.values())
        for term, count in filtered_terms.items():
            j = features_dict[term]
            if not args.tf:
                all_doc_features[i, j] = 1
            else:
                all_doc_features[i, j] = count / num_of_terms
    # Then, if `args.idf` is set, multiply the document features by the (IDF),using the variant which contains `+1` in the denominator;
    # The IDFs are computed on the train set and then reused without modification on the test set.
    if args.idf:
        return all_doc_features * idf
    return all_doc_features


def main(args: argparse.Namespace) -> float:
    newsgroups = NewsGroups(data_size=args.train_size + args.test_size)

    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        newsgroups.data, newsgroups.target, test_size=args.test_size, random_state=RANDOM_SEED)

    # Createing a feature for every term that is present at least twice
    # in the training data. A term is every maximal sequence of at least 1 word character,
    # where a word character corresponds to a regular expression `\w`.
    # How many documents contain this term
    term_counter = collections.Counter()
    documents_with_term = collections.defaultdict(int)
    for doc  in train_data:
        terms = re.findall(r'\w+', doc)
        term_counter.update(terms)
        no_dup_terms = set(terms)
        for term in no_dup_terms:
            documents_with_term[term]+=1
    features = np.array([term for term, count in term_counter.items() if count > 1])

    numerator = np.log(len(train_data))
    idf = np.full(len(features), numerator)
    for i,feature in enumerate(features):
        idf[i] -= np.log(documents_with_term[feature]+1)

    train_features = get_tf_idf(args,train_data,features,idf)
    test_features = get_tf_idf(args,test_data,features,idf)

    model = sklearn.linear_model.LogisticRegression(solver="liblinear", C=10_000)
    model.fit(train_features, train_target)
    test_predictions = model.predict(test_features)
    
    f1_score = sklearn.metrics.f1_score(test_target, test_predictions, average="macro")
    return 100 * f1_score


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    f1_score = main(main_args)
    print("F-1 score for TF={}, IDF={}: {:.1f}%".format(main_args.tf, main_args.idf, f1_score))
