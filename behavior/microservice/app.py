import pickle
import pandas as pd
from statistics import mean, median
from pathlib import Path
from typing import Dict

import sqlite3
import numpy as np
import setproctitle
from flask import Flask, render_template, jsonify, request, redirect, url_for, g
from plumbum import cli
import threading

from behavior import DIFFED_TARGET_COLS
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
    Y = dict(zip(DIFFED_TARGET_COLS, Y))
    return jsonify(Y)


def connect():
    db = getattr(g, '__database', None)
    if db is None:
        db = g.__database = sqlite3.connect('inference.db')
    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '__database', None)
    if db is not None:
        db.close()


def init_db():
    with app.app_context():
        db = connect()
        db.cursor().execute('''
            CREATE TABLE IF NOT EXISTS inference_results (
                query TEXT,
                predicted_cost REAL,
                true_cost REAL,
                true_cost_valid INT,
                action_state TEXT,
                predicted_results TEXT)
        ''')
        db.commit()


def get_inference_results(query):
    # This by default returns the data sorted by query_id, then ordered
    # by situations where true costing error'ed, and finally ordered by
    # abs(true_cost - predicted_cost).
    #
    # Ordering by abs(true_cost - predicted_cost) gives us the instances
    # where our prediction is way off first.
    query_clause = "ORDER BY query, true_cost_valid, ABS(true_cost - predicted_Cost) DESC"
    if query is not None:
        query_clause = query

    def dict_factory(cursor, row):
        d = {}
        for idx, col in enumerate(cursor.description):
            d[col[0]] = row[idx]
        return d

    db = connect()
    db.row_factory = dict_factory
    cursor = db.cursor().execute('SELECT *, ABS(true_cost - predicted_cost) as cost_diff FROM inference_results ' + query_clause)
    result_set = cursor.fetchall()
    db.commit()
    return result_set


@app.route("/")
def index():
    results = get_inference_results(None)
    return render_template('index.html', title='Inference Results', results=results)


@app.route("/prediction_results", methods=["GET", "POST", "DELETE"])
def prediction_results():
    if request.method == 'POST':
        try:
            query = request.form['query']
            predicted_cost = float(request.form['predicted_cost'])
            true_cost_valid = request.form['true_cost_valid'] == '1'
            true_cost = float(request.form['true_cost'])
            predicted_results = request.form['predicted_results']
            action_state = request.form['action_state']
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
        db.cursor().execute(insert_stmt, (query, predicted_cost, true_cost, true_cost_valid, action_state, predicted_results))
        db.commit()
        return "", 200

    elif request.method == 'GET':
        query = request.args['query'] if 'query' in request.args else None
        result_set = get_inference_results(query)
        return jsonify(result_set)

    elif request.method == 'DELETE':
        db = connect()
        db.cursor().execute('DELETE FROM inference_results')
        db.commit()
        return "", 200


class ModelMicroserviceCLI(cli.Application):
    models_path = cli.SwitchAttr("--models-path", Path, mandatory=True)

    def main(self, *args):
        init_db()
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
