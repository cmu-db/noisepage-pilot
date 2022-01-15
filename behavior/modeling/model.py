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

from behavior.modeling import METHODS


def get_model(method, config):
    """Initialize and return the underlying Behavior Model variant with the provided configuration parameters.

    Parameters
    ----------
    method : str
        Regression model variant.
    config : dict
        Configuration parameters for the model.

    Returns
    -------
    Any
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
    if method == "rf":
        regressor = RandomForestRegressor(
            n_estimators=config["rf"]["n_estimators"],
            criterion=config["rf"]["criterion"],
            max_depth=config["rf"]["max_depth"],
            random_state=config["random_state"],
            n_jobs=config["num_jobs"],
        )
    if method == "gbm":
        regressor = LGBMRegressor(
            max_depth=config["gbm"]["max_depth"],
            num_leaves=config["gbm"]["num_leaves"],
            n_estimators=config["gbm"]["n_estimators"],
            min_child_samples=config["gbm"]["min_child_samples"],
            objective=config["gbm"]["objective"],
            random_state=config["random_state"],
        )
        regressor = MultiOutputRegressor(regressor)
    # Multi-layer Perceptron.
    if method == "mlp":
        hls = tuple(dim for dim in config["mlp"]["hidden_layers"])
        regressor = MLPRegressor(
            hidden_layer_sizes=hls,
            early_stopping=config["mlp"]["early_stopping"],
            max_iter=config["mlp"]["max_iter"],
            alpha=config["mlp"]["alpha"],
            random_state=config["random_state"],
        )
    # Generalized Linear Models.
    if method == "lr":
        regressor = LinearRegression(n_jobs=config["num_jobs"])
    if method == "huber":
        regressor = HuberRegressor(max_iter=config["huber"]["max_iter"])
        regressor = MultiOutputRegressor(regressor)
    if method == "mt_lasso":
        regressor = MultiTaskLasso(alpha=config["mt_lasso"]["alpha"], random_state=config["random_state"])
    if method == "lasso":
        regressor = Lasso(alpha=config["lasso"]["alpha"], random_state=config["random_state"])
    if method == "elastic":
        regressor = ElasticNet(
            alpha=config["elastic"]["alpha"],
            l1_ratio=config["elastic"]["l1_ratio"],
            random_state=config["random_state"],
        )
        regressor = MultiOutputRegressor(regressor)
    if method == "mt_elastic":
        regressor = MultiTaskElasticNet(l1_ratio=config["mt_elastic"]["l1_ratio"], random_state=config["random_state"])

    return regressor


class BehaviorModel:
    def __init__(self, output_dir, method, ou_name, timestamp, config, features):
        """Create a Behavior Model for predicting the resource consumption cost of a single Postgres operating-unit.

        Parameters
        ----------
        output_dir : [type]
            [description]
        method : str
            [description]
        ou_name : str
            [description]
        timestamp : str
            [description]
        config : dict[str, Any]
            [description]
        features : list[str]
            [description]
        """
        self.output_dir = output_dir
        self.method = method
        self.timestamp = timestamp
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

    def save(self):
        """Save the model to disk."""
        model_dir = self.output_dir / self.timestamp / self.method / self.ou_name
        with open(model_dir / f"{self.method}_{self.ou_name}.pkl", "wb") as f:
            pickle.dump(self.model, f)
