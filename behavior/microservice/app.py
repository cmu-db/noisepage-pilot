import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import setproctitle
from flask import Flask, jsonify, request
from plumbum import cli

from behavior.modeling.model import BehaviorModel

app = Flask(__name__)
app.config["DEBUG"] = True


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
    Y = dict(zip(behavior_model.targets, Y))
    return jsonify(Y)


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
