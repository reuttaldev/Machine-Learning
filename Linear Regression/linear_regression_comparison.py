import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt
import sklearn.linear_model
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.compose

parser = argparse.ArgumentParser()
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--model_path", default="rental_competition.model", type=str, help="Model path")
RANDOM_SEED = 42


class Dataset:
    """Dataset features:
    - season (1: winter, 2: spring, 3: summer, 4: autumn)
    - year (0: 2011, 1: 2012)
    - month (1-12)
    - hour (0-23)
    - holiday (binary indicator)
    - day of week (0: Sun, 1: Mon, ..., 6: Sat)
    - working day (binary indicator; a day is neither weekend nor holiday)
    - weather (1: clear, 2: mist, 3: light rain, 4: heavy rain)
    - temperature (normalized so that -8 Celsius is 0 and 39 Celsius is 1)
    - feeling temperature (normalized so that -16 Celsius is 0 and 50 Celsius is 1)
    - relative humidity (0-1 range)
    - windspeed (normalized to 0-1 range)

    The target variable is the number of rented bikes in the given hour.
    """
    
    def __init__(self,
                 name="rental_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        
        for key, value in dataset.items():
            setattr(self, key, value)

models_to_check = {
    'LinearRegression': sklearn.linear_model.LinearRegression(),
    'Ridge': sklearn.linear_model.Ridge(),
    'SGDRegressor': sklearn.linear_model.SGDRegressor(),
    'Lasso': sklearn.linear_model.Lasso(),
}
params_to_check = {
    'Ridge': {'alpha':np.logspace(-3, 0, 4)},
    'SGDRegressor': {
        'loss': ['squared_error'],
        'alpha': np.logspace(-3, 0, 4),
        'learning_rate': ['invscaling', 'constant'],
        'eta0': [0.001, 0.01, 0.1, 1, 10],
        'max_iter': [5000,7000],
        'penalty': ['l2', 'l1']
    },
    'Lasso': {'alpha': np.logspace(-3, 0, 4)}
}

class MyModel:
    def  __init__(self):
        dataset = Dataset()
        self.X_train = dataset.data
        self.t_train = dataset.target
        self.cv_folds = 5
        self.preprocessor = self.preprocess_features()
        self.X_train= self.preprocessor.fit_transform(self.X_train)
        self.sorted_trained_models = self.train_and_find_best_models()

    def get_columns_indices(self,matrix,integer):
        l = []
        for i in range(matrix.shape[1]):      
            column = matrix[:,i]  
            # if all elements in the column are integers
            # 3.0 is considered int, that's why there's the second condition
            if np.issubdtype(column.dtype, np.integer) or np.all(column== column.astype(int)):
                if integer:
                    l.append(i)
            elif not integer:
                l.append(i)
        return l  
    
    # If an input column has only integer values, consider it a categorical column -- encode with one-hot encoding
    # using `handle_unknown="ignore"` to ignore missing values in test set).
    # Otherwise, normalize their values s.t mean 0 and variance 1 (to ensure that all features contribute equally to model training)
    def preprocess_features(self):
        cat_columns_indices = self.get_columns_indices(self.X_train, True)
        one_hot_transformer = sklearn.preprocessing.OneHotEncoder(sparse_output=True, handle_unknown="ignore")
        real_columns_indices = self.get_columns_indices(self.X_train, False)
        normalize_transformer = sklearn.preprocessing.StandardScaler()
        preprocess_transformer= sklearn.compose.ColumnTransformer([
        ("categorical",one_hot_transformer,cat_columns_indices),
        ("real-value",normalize_transformer,real_columns_indices)])
        # Appending polynomial features of order 2 -- i.e. if the input values are `[a, b, c, d]`
        # append  `[a^2, ab, ac, ad, b^2, bc, bd, c^2, cd, d^2]`. 
        #Creating polynomial features can help capture non-linear relationships in your dataset,
        poly_transformer = sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)
        pipeline = sklearn.pipeline.Pipeline([
            ("preprocessing", preprocess_transformer),
            ("polynomial", poly_transformer)])
        return pipeline

    def train_and_find_best_models(self):
        models = {}
        scores = {}
        best_params = {}
        best_score = float('inf')

        for model_name, model in models_to_check.items():            
            if model_name in params_to_check: 
                grid_search = sklearn.model_selection.GridSearchCV(
                    model, 
                    params_to_check[model_name], 
                    cv=self.cv_folds, 
                    scoring='neg_root_mean_squared_error'
                )
                grid_search.fit(self.X_train, self.t_train)
                models[model_name] = grid_search.best_estimator_
                scores[model_name] = -grid_search.best_score_ 
                best_params[model_name] = grid_search.best_params_
                
            else:
                model.fit(self.X_train, self.t_train)
                models[model_name] = model
                cv_scores = sklearn.model_selection.cross_val_score(
                    model, 
                    self.X_train, 
                    self.t_train, 
                    cv=self.cv_folds, 
                    scoring='neg_root_mean_squared_error'
                )
                scores[model_name] = -np.mean(cv_scores)

            if scores[model_name] < best_score:
                best_score = scores[model_name]

        sorted_keys = sorted(scores, key=scores.get)  # ascending RMSE
        print("Training scores")
        for rank, name in enumerate(sorted_keys, start=1):

            print(f"{rank}. {name} RMSE = {scores[name]:.4f}")
            if name in best_params:
                print(f"best params: {best_params[name]}")

        sorted_models = [models[name] for name in sorted_keys]
        return sorted_models
    
    def predict(self,X_test):
        if self.sorted_trained_models == None:
            return
        X_test = self.preprocessor.transform(X_test)
        # aggregate predictions 
        predictions = np.zeros(X_test.shape[0])
        # calculate weights based on the index (1-based)
        weights = [0.6,0.25,0.15,0] 
        for weight, model in zip(weights, self.sorted_trained_models):
            predictions+= model.predict(X_test) * weight
        return predictions 
    
def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        np.random.seed(RANDOM_SEED)

        my_model = MyModel()
        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(my_model, model_file)

    else:

        with lzma.open(args.model_path, "rb") as model_file:
            my_model = pickle.load(model_file)

        test = Dataset(args.predict).data
        predictions = my_model.predict(test)
        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
