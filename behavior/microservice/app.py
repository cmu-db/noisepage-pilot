import pickle
import sqlite3
import time
from pathlib import Path
from typing import Dict

import flask
import numpy as np
import setproctitle
from flask import Flask, g, jsonify, render_template, request, send_from_directory
from plumbum import cli

from behavior import DIFFED_TARGET_COLS
from behavior.modeling.model import BehaviorModel

app = Flask(__name__, static_url_path="")
app.config["DEBUG"] = True


@app.route("/static/<path:path>")
def send_static(path):
    return send_from_directory("static", path)


def _infer_model(model_type, ou_type, features, jsonify_result):
    # Get the behavior model.
    try:
        behavior_model: BehaviorModel = app.config["model_map"][model_type][ou_type]
    except KeyError as err:
        return f"Error cannot find {model_type} model: {err}"

    # Check that all the features are present.
    diff = set(behavior_model.features).difference(features)
    if len(diff) > 0:
        return f"{model_type}:{ou_type} Features missing: {diff}"

    # Extract the features.
    X = [features[feature] for feature in behavior_model.features]
    X = np.array(X).astype(float).reshape(1, -1)

    # Predict the Y values.
    start = time.time()
    Y = behavior_model.predict(X)
    assert Y.shape[0] == 1
    Y = Y[0]
    end = time.time()

    Y = dict(zip(DIFFED_TARGET_COLS, Y))

    # Modify Y so that we also accountf for inference_time, Model_type, and ou_type
    Y["model_type"] = model_type
    Y["ou_type"] = ou_type
    Y["inference_time"] = end - start
    return jsonify(Y) if jsonify_result else Y


@app.route("/model/<model_type>/<ou_type>/", methods=["GET"])
def infer(model_type, ou_type):
    return _infer_model(model_type, ou_type, request.args, jsonify_result=True)


@app.route("/batch_infer", methods=["POST"])
def batch_infer():
    json_data = flask.request.json
    infer_results = []
    for infer_request in json_data:
        model_type = infer_request["model_type"]
        ou_type = infer_request["ou_type"]
        features = infer_request["features"]
        infer_results.append(_infer_model(model_type, ou_type, features, jsonify_result=False))
    return jsonify(infer_results)


def connect():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect("./artifacts/behavior/microservice/inference.db")
    return db


@app.teardown_appcontext
def close_connection(_):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


def _init_db():
    with app.app_context():
        db = connect()
        db.cursor().execute(
            """
            CREATE TABLE IF NOT EXISTS inference_results (
                query TEXT,
                predicted_cost REAL,
                true_cost REAL,
                true_cost_valid INT,
                action_state TEXT,
                predicted_results TEXT)
        """
        )
        db.commit()


def _get_inference_results(query):
    """
    Function to get the inference results from the sqlite database instance.

    Parameters
    ----------
    query: str
        The query to execute against the sqlite database instance.

    Returns
    -------
    result_set: list[dict]
        A list of row dictionaries. Each element in the list represents a tuple from
        the database in a key-value dictionary access format.
    """

    # https://docs.python.org/3/library/sqlite3.html#sqlite3.Connection.row_factory
    def dict_factory(cursor, row):
        row_dict = {}
        for idx, col in enumerate(cursor.description):
            col_name = col[0]
            row_dict[col_name] = row[idx]
        return row_dict

    db = connect()
    db.row_factory = dict_factory
    cursor = db.cursor().execute(query)
    result_set = cursor.fetchall()
    db.commit()
    return result_set


@app.route("/")
def index():
    results = _get_inference_results(
        """
        SELECT *, ABS(true_cost - predicted_cost) as cost_diff
        FROM inference_results
        ORDER BY query, true_cost_valid, ABS(true_cost - predicted_cost) DESC
        """
    )

    return render_template("index.html", title="Inference Results", results=results)


@app.route("/prediction_results", methods=["GET", "POST", "DELETE"])
def prediction_results():
    if request.method == "POST":
        db = connect()
        insert_stmt = """
            INSERT INTO inference_results (
                query,
                predicted_cost,
                true_cost,
                true_cost_valid,
                action_state,
                predicted_results)
            VALUES (?, ?, ?, ?, ?, ?)
        """

        try:
            results = flask.request.json
            for result in results:
                query = result["query"]
                predicted_cost = result["predicted_cost"]
                true_cost_valid = result["true_cost_valid"] == 1
                true_cost = result["true_cost"]
                predicted_results = result["predicted_results"]
                action_state = result["action_state"]

                db.cursor().execute(
                    insert_stmt, (query, predicted_cost, true_cost, true_cost_valid, action_state, predicted_results)
                )
        except KeyError as err:
            return f"KeyError: {err}", 400
        except ValueError as err:
            return f"ValueError: {err}", 400
        finally:
            db.commit()

        return "", 200

    if request.method == "GET":
        query = request.args["query"] if "query" in request.args else None
        result_set = _get_inference_results(query)
        return jsonify(result_set)

    if request.method == "DELETE":
        db = connect()
        db.cursor().execute("DELETE FROM inference_results")
        db.commit()
        return "", 200

    return "", 405


class ModelMicroserviceCLI(cli.Application):
    models_path = cli.SwitchAttr("--models-path", Path, mandatory=True)

    def main(self, *args):
        _init_db()
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
