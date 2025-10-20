# Linear regression

## Analytical solution
In [explicit_linear_regression.py](Linear%20Regression/explicit_linear_regression.py), I compute the analytical (closed-form) solution for the optimal model weights in linear regression.  
The optimal weights formula is obtained by setting the derivative of the **mean squared error (MSE)** loss function to zero, with respect to all weights:  
$\boldsymbol{w} = (X^\top X)^{-1} X^\top y$

**Example usage:**  
`python "Linear Regression/explicit_linear_regression.py" --test_size=0.1`  
**Example output:**  
`52.38`

## Stochastic gradient descent
In [linear_regression_sgd.py](Linear Regression\logistic_regression_sgd.py) I approximate the optimal weights using mini-batch stochastic gradient descent (SGD) with L2 regularization.
The loss function minimized is one half of the mean squared error (error function).  
**Example usage:**  
`python "Linear Regression\sgd_linear_regression.py" --batch_size=10 --epochs=50 --learning_rate=0.01`  
**Example output:**  
```
Test RMSE: SGD 90.958, explicit 91.5
Learned weights: 3.944 7.517 0.084 30.820 -1.721 -1.129 -1.980 6.294 1.980 -10.597 -13.841 -4.312 ...
```

## L2 regression
In [ridge_regression.py](Linear%20Regression/ridge_regression.py), I evaluate Ridge Regression — a type of linear regression that adds an L2 regularization term to the loss function in order to prevent overfitting and improve model generalization. The model minimizes the following objective:  

$J(\mathbf{w}) = \|y - X\mathbf{w}\|^2 + \lambda \|\mathbf{w}\|^2$

It evaluates **500 different λ (lambda)** values, geometrically spaced between 0.01 and 10, and return the lambda producing lowest one and the corresponding value.

**Example usage:**  
`python "Linear Regression\l2_linear_regression.py" --test_size=0.15 --plot`  
**Example output:** 
`0.49 52.11`
![RMSE vs λ example][figures\l2_linear_regression_figure.png]

## Grid search
In [linear_regression_comparison.py](Linear%20Regression/linear_regression_comparison.py) I evaluate multiple linear regression–based models to predict hourly bike rental demand based on data from a bike rental shop.
### Implementation 
#### 1. **Feature Preprocessing**
- **Categorical columns:** automatically detected (integer-only columns) and one-hot encoded using `OneHotEncoder(handle_unknown="ignore")`.
- **Real-valued columns:** scaled using `StandardScaler` for zero mean and unit variance.
- Combined using `ColumnTransformer`.

#### 2. **Polynomial Feature Expansion**
Adds **2nd-order polynomial features** to capture non-linear relationships and feature interactions.

#### 3. **Model Comparison**
The following models from sklearn are trained and evaluated using in order to find the one that minimizes the loss function. Cross-validation uses negative RMSE as the scoring metric.  
Additionally, I used `GridSearchCV` finds the best hyperparameters for each model.

| Model | Regularization | Parameters Tuned |
|--------|----------------|------------------|
| `LinearRegression` | None | – |
| `Ridge` | L2 | `alpha ∈ {10⁻³, 10⁻², 10⁻¹, 1}` |
| `Lasso` | L1 | `alpha ∈ {10⁻³, 10⁻², 10⁻¹, 1}` |
| `SGDRegressor` | L1 or L2 | `alpha`, `eta0`, `learning_rate`, `penalty`, `max_iter` |

#### 4. **Model Ranking and Ensemble**
- All models are scored by mean RMSE.
- Models are sorted by performance.
- The top 3 models are **ensembled** with weighted averaging:

**Example usage:**  
`python "Linear Regression\linear_regression_comparison.py"`
**Example output:**  
```
Training scores
1. Lasso RMSE = 66.4811
best params: {'alpha': 0.1}
2. SGDRegressor RMSE = 67.2270
best params: {'alpha': 0.1, 'eta0': 0.001, 'learning_rate': 'invscaling', 'loss': 'squared_error', 'max_iter': 7000, 'penalty': 'l1'}
3. Ridge RMSE = 75.0376
best params: {'alpha': 1.0}
4. LinearRegression RMSE = 215.5452
```

The RMSE for some training data were 64.76.

# Logistic regression