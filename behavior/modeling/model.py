from __future__ import annotations

import pickle

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    MultiTaskElasticNet,
    MultiTaskLasso,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from behavior.modeling import METHODS, featurize


def get_model(method, config):
    """Initialize and return the underlying Behavior Model variant with the provided configuration parameters.

    Parameters
    ----------
    method : str
        Regression model variant.
    config : dict[str, Any]
        Configuration parameters for the model.

    Returns
    -------
    model : Any
       A regression model.

    Raises
    ------
    ValueError
        If the requested method is not supported.
    """
    if method not in METHODS:
        raise ValueError(f"Method: {method} is not supported.")

    regressor = None

    # Tree-based Models.
    if method == "dt":
        regressor = DecisionTreeRegressor(max_depth=config["dt"]["max_depth"])
        regressor = MultiOutputRegressor(regressor)
    elif method == "rf":
        regressor = RandomForestRegressor(
            n_estimators=config["rf"]["n_estimators"],
            criterion=config["rf"]["criterion"],
            max_depth=config["rf"]["max_depth"],
            random_state=config["random_state"],
            n_jobs=config["num_jobs"],
        )
    elif method == "gbm":
        regressor = LGBMRegressor(
            max_depth=config["gbm"]["max_depth"],
            num_leaves=config["gbm"]["num_leaves"],
            n_estimators=config["gbm"]["n_estimators"],
            min_child_samples=config["gbm"]["min_child_samples"],
            objective=config["gbm"]["objective"],
            random_state=config["random_state"],
        )
        regressor = MultiOutputRegressor(regressor)
    elif method == "mlp":
        # Multi-layer Perceptron.
        hls = tuple(dim for dim in config["mlp"]["hidden_layers"])
        regressor = MLPRegressor(
            hidden_layer_sizes=hls,
            early_stopping=config["mlp"]["early_stopping"],
            max_iter=config["mlp"]["max_iter"],
            alpha=config["mlp"]["alpha"],
            random_state=config["random_state"],
        )
    # Generalized Linear Models.
    elif method == "lr":
        regressor = LinearRegression(n_jobs=config["num_jobs"])
    elif method == "huber":
        regressor = HuberRegressor(max_iter=config["huber"]["max_iter"])
        regressor = MultiOutputRegressor(regressor)
    elif method == "mt_lasso":
        regressor = MultiTaskLasso(alpha=config["mt_lasso"]["alpha"], random_state=config["random_state"])
    elif method == "lasso":
        regressor = Lasso(alpha=config["lasso"]["alpha"], random_state=config["random_state"])
    elif method == "elastic":
        regressor = ElasticNet(
            alpha=config["elastic"]["alpha"],
            l1_ratio=config["elastic"]["l1_ratio"],
            random_state=config["random_state"],
        )
        regressor = MultiOutputRegressor(regressor)
    elif method == "mt_elastic":
        regressor = MultiTaskElasticNet(l1_ratio=config["mt_elastic"]["l1_ratio"], random_state=config["random_state"])

    assert regressor is not None
    return regressor


class BehaviorModel:
    def __init__(self, method, ou_name, base_model_name, config, features):
        """Create a Behavior Model for predicting the resource consumption cost of a single PostgreSQL operating-unit.

        Parameters
        ----------
        method : str
            The method to use. Valid methods are defined in modeling/__init__.py.
        ou_name : str
            The name of this operating unit.
        base_model_name : str
            The base name for this model, currently just the experiment name.
        config : dict[str, Any]
            The dictionary of configuration parameters for this model.
        features : list[str]
            Metadata describing input features for this model.
        """
        self.method = method
        self.base_model_name = base_model_name
        self.ou_name = ou_name
        self.model = get_model(method, config)
        self.features = features
        self.normalize = config["normalize"]
        self.log_transform = config["log_transform"]
        self.eps = 1e-4
        self.xscaler = RobustScaler() if config["robust"] else StandardScaler()
        self.yscaler = RobustScaler() if config["robust"] else StandardScaler()

    def train(self, x, y):
        """Train a model using the input features and targets.

        Parameters
        ----------
        x : NDArray[np.float32]
            Input features.
        y : NDArray[np.float32]
            Input targets.
        """
        if self.log_transform:
            x = np.log(x + self.eps)
            y = np.log(y + self.eps)

        if self.normalize:
            x = self.xscaler.fit_transform(x)
            y = self.yscaler.fit_transform(y)

        self.model.fit(x, y)

    def predict(self, x):
        """Run inference using the provided input features.

        Parameters
        ----------
        x : NDArray[np.float32]
            Input features.

        Returns
        -------
        NDArray[np.float32]
            Predicted targets.
        """
        # Transform the features.
        if self.log_transform:
            x = np.log(x + self.eps)
        if self.normalize:
            x = self.xscaler.transform(x)

        # Perform inference (in the transformed feature space).
        y = self.model.predict(x)

        # Map the result back to the original space.
        if self.normalize:
            y = self.yscaler.inverse_transform(y)
        if self.log_transform:
            y = np.exp(y) - self.eps
            y = np.clip(y, 0, None)

        return y

    def convert_raw_input(self, X):
        """
        Given X which is a raw input feature vector, this function returns X'
        by applying the feature configuration contained in self.features. X'
        can be passed to predict.

        Parameters
        ----------
        X : pandas.DataFrame
            Input DataFrame of X's to predict the Y's for.

        Returns
        -------
        X' : pandas.DataFrame
            Transformed X' that can be used as inputs to the model.
        """
        return featurize.extract_input_features(X, self.features)

    def save(self, output_dir):
        """Save the model to disk.

        Parameters
        ----------
        output_dir : Path | str
            The directory to save the model to.
        """
        model_dir = output_dir / self.base_model_name / self.method / self.ou_name
        with open(model_dir / f"{self.method}_{self.ou_name}.pkl", "wb") as f:
            pickle.dump(self, f)
