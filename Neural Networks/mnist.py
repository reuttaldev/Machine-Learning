import argparse
import lzma
import pickle
import sys
from typing import Optional

import sys
sys.path.append(".")
from config import RANDOM_SEED

import numpy as np
import numpy.typing as npt
import sklearn.preprocessing
import sklearn.neural_network
import sklearn.pipeline
import sklearn.ensemble
from sklearn.metrics import accuracy_score

from tensorflow.keras.datasets import mnist

parser = argparse.ArgumentParser()
parser.add_argument("--predict", action="store_true")
parser.add_argument("--model_path", default="Neural Networks\mnist.model", type=str, help="Model path")


class MyModel:
    def  __init__(self,X_train, y_train):
        self.X_train = X_train.reshape([-1, 28*28]).astype(float)
        self.t_train = y_train
        self.pipline = sklearn.pipeline.Pipeline([
            ("preprocesser",self.get_preprocessor()),
            ("ensemble", self.get_ensamble(5))
        ])
        self.pipline.fit(self.X_train,self.t_train)

    
    def get_preprocessor(self):
        return sklearn.preprocessing.MinMaxScaler()
    
    def get_model(self):
        return sklearn.neural_network.MLPClassifier(hidden_layer_sizes=500, max_iter=15, verbose=1)

    def get_ensamble(self, count):
        return sklearn.ensemble.VotingClassifier(
            #name, model tuple in a list of all models
            [(f"model{i}", self.get_model()) for i in range(count)],
              voting="soft")
    
def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    if args.predict is False:
        # We are training a model.
        np.random.seed(RANDOM_SEED)
        my_model = MyModel(X_train, y_train)
        model = my_model.pipline
        # If you trained one or more MLPs, you can use the following code
        # to compress it significantly (approximately 12 times). The snippet
        # assumes the trained `MLPClassifier` is in the `mlp` variable.
        for mlp in model["ensemble"].estimators_:
            mlp._optimizer = None
            for i in range(len(mlp.coefs_)): mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
            for i in range(len(mlp.intercepts_)): mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)
        test = X_test.reshape([-1, 28*28]).astype(float)

        predictions = model.predict(test)
        print("Test accuracy:", accuracy_score(y_test, predictions))

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)