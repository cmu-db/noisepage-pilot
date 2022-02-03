import pickle
import pandas as pd
from statistics import mean, median
from pathlib import Path
from typing import Dict

import numpy as np
import setproctitle
from flask import Flask, render_template, jsonify, request, redirect, url_for
from plumbum import cli
import threading

from behavior import DIFFED_TARGET_COLS
from behavior.modeling.model import BehaviorModel

app = Flask(__name__)
app.config["DEBUG"] = True

# These results are protected with the results_lock. This model works in development
# but does not work with a wsgi interface (we assume there is only 1 process). This
# does not survive restarts.
results_lock = threading.Lock()
results = []


@app.route("/")
def index():
    return redirect(url_for("prediction_results"))


@app.route("/model/<model_type>/<ou_type>/", methods=["GET"])
def infer(model_type, ou_type):
    # Get the behavior model.
    try:
        behavior_model: BehaviorModel = app.config["model_map"][model_type][ou_type]
    except KeyError as err:
        return f"Error: {err}"

    # Check that all the features are present.
    diff = set(behavior_model.features).difference(request.args)
    if len(diff) > 0:
        return f"Features missing: {diff}"

    # Extract the features.
    X = [request.args[feature] for feature in behavior_model.features]
    X = np.array(X).astype(float).reshape(1, -1)

    # Predict the Y values.
    Y = behavior_model.predict(X)
    assert Y.shape[0] == 1
    Y = Y[0]

    # Label and return the Y values.
    Y = dict(zip(DIFFED_TARGET_COLS, Y))
    return jsonify(Y)


@app.route("/prediction_results", methods=["GET", "POST", "DELETE"])
def prediction_results():
    global results_lock
    global results
    if request.method == 'POST':
        results_lock.acquire()
        results.append(request.form)
        results_lock.release()
        return "", 200

    elif request.method == 'GET':
        results_lock.acquire()
        results_serialized = jsonify(results)
        results_lock.release()
        return results_serialized

    elif request.method == 'DELETE':
        results_lock.acquire()
        results = []
        results_lock.release()
        return "", 200


class ModelMicroserviceCLI(cli.Application):
    models_path = cli.SwitchAttr("--models-path", Path, mandatory=True)

    def main(self, *args):
        self.models_path = self.models_path.absolute()

        model_map: Dict[str, Dict[str, BehaviorModel]] = {}
        # Load all the models in memory for inference.
        for model_type_path in self.models_path.glob("*"):
            model_type = model_type_path.name
            model_map[model_type] = model_map.get(model_type, {})
            for ou_type_path in model_type_path.glob("*"):
                ou_type = ou_type_path.name

                model_path = list(ou_type_path.glob("*.pkl"))
                assert len(model_path) == 1
                model_path = model_path[0]

                print(model_type, ou_type, model_path)
                with open(model_path, "rb") as model_file:
                    model = pickle.load(model_file)
                    model_map[model_type][ou_type] = model

        # Expose the models to the Flask app.
        app.config["model_map"] = model_map
        setproctitle.setproctitle("Behavior Models Microservice")
        # Run the Flask app.
        app.run()


if __name__ == "__main__":
    ModelMicroserviceCLI.run()
