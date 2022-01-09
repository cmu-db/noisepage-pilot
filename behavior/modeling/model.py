from __future__ import annotations

import pickle
from typing import Any

import numpy as np
from lightgbm import LGBMRegressor
from numpy.typing import NDArray
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

from behavior import MODEL_DATA_DIR
from behavior.modeling import METHODS


def get_model(method: str, config: dict[str, Any]) -> Any:
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")

    regressor = None
    if method == "lr":
        regressor = LinearRegression(n_jobs=config["num_jobs"])
    if method == "huber":
        regressor = HuberRegressor(max_iter=config["huber"]["max_iter"])
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
            random_state=config["random_state"],
        )
        regressor = MultiOutputRegressor(regressor)
    if method == "mlp":
        hls = tuple(dim for dim in config["mlp"]["hidden_layers"])
        regressor = MLPRegressor(
            hidden_layer_sizes=hls,
            early_stopping=config["mlp"]["early_stopping"],
            max_iter=config["mlp"]["max_iter"],
            alpha=config["mlp"]["alpha"],
        )
    if method == "mt_lasso":
        regressor = MultiTaskLasso(alpha=config["mt_lasso"]["alpha"])
    if method == "lasso":
        regressor = Lasso(alpha=config["lasso"]["alpha"])
    if method == "dt":
        regressor = DecisionTreeRegressor(max_depth=config["dt"]["max_depth"])
        regressor = MultiOutputRegressor(regressor)
    if method == "elastic":
        regressor = ElasticNet(
            alpha=config["elastic"]["alpha"], l1_ratio=config["elastic"]["l1_ratio"]
        )
        regressor = MultiOutputRegressor(regressor)
    if method == "mt_elastic":
        regressor = MultiTaskElasticNet(l1_ratio=config["mt_elastic"]["l1_ratio"])

    return regressor


class BehaviorModel:
    def __init__(
        self,
        method: str,
        ou_name: str,
        timestamp: str,
        config: dict[str, Any],
        features: list[str],
        targets: list[str],
    ):
        self.method = method
        self.timestamp = timestamp
        self.ou_name = ou_name
        self.model = get_model(method, config)
        self.features = features
        self.targets = targets
        self.normalize = config["normalize"]
        self.log_transform = config["log_transform"]
        self.eps = 1e-4
        self.xscaler = RobustScaler() if config["robust"] else StandardScaler()
        self.yscaler = RobustScaler() if config["robust"] else StandardScaler()

    def train(self, x: NDArray[np.float32], y: NDArray[np.float32]) -> None:
        if self.log_transform:
            x = np.log(x + self.eps)
            y = np.log(y + self.eps)

        if self.normalize:
            x = self.xscaler.fit_transform(x)
            y = self.yscaler.fit_transform(y)

        self.model.fit(x, y)

    def predict(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        # transform the features
        if self.log_transform:
            x = np.log(x + self.eps)
        if self.normalize:
            x = self.xscaler.transform(x)

        # make prediction
        y: NDArray[np.float32] = self.model.predict(x)

        # transform the y back
        if self.normalize:
            y = self.yscaler.inverse_transform(y)
        if self.log_transform:
            y = np.exp(y) - self.eps
            y = np.clip(y, 0, None)

        return y

    def save(self) -> None:
        model_dir = MODEL_DATA_DIR / self.timestamp / self.method / self.ou_name
        with open(model_dir / f"{self.method}_{self.ou_name}.pkl", "wb") as f:
            pickle.dump(self.model, f)
