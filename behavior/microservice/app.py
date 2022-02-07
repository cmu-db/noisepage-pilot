import pickle
import sqlite3
from pathlib import Path
from typing import Dict

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


def connect():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect("inference.db")
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
        try:
            query = request.form["query"]
            predicted_cost = float(request.form["predicted_cost"])
            true_cost_valid = request.form["true_cost_valid"] == "1"
            true_cost = float(request.form["true_cost"])
            predicted_results = request.form["predicted_results"]
            action_state = request.form["action_state"]
        except KeyError as err:
            return f"KeyError: {err}", 400
        except ValueError as err:
            return f"ValueError: {err}", 400

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
        db.cursor().execute(
            insert_stmt, (query, predicted_cost, true_cost, true_cost_valid, action_state, predicted_results)
        )
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
