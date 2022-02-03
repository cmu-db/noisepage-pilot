from __future__ import annotations

import itertools
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from pandas import DataFrame
from plumbum import cli
from pydotplus import graphviz
from sklearn import tree
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from behavior import (
    BASE_TARGET_COLS,
    BENCHDB_TO_TABLES,
    DIFFED_TARGET_COLS,
    PLAN_NODE_NAMES,
)
from behavior.modeling.model import BehaviorModel

logger = logging.getLogger(__name__)


def evaluate(model, df, output_dir, dataset, mode):
    """Evaluate the model.

    Parameters
    ----------
    ou_model : BehaviorModel
        Model to evaluate.
    df : DataFrame
        Evaluation data.
    output_dir : Path
        Results output directory.
    dataset : str
        Benchmark name.
    mode : str
        Training or evaluation.

    Raises
    ------
    ValueError
        If the caller passed an invalid mode.
    """
    if mode not in ["train", "eval"]:
        raise ValueError(f"Invalid mode: {mode}")

    # Split the features and targets.
    X = df[model.features].values
    y = df[DIFFED_TARGET_COLS].values

    # Run inference.
    y_pred = model.predict(X)

    # Pair and re-order the target columns for more readable outputs.
    pred_cols = [f"pred_{col}" for col in DIFFED_TARGET_COLS]
    paired_cols = zip(pred_cols, DIFFED_TARGET_COLS)
    reordered_cols = model.features + list(itertools.chain.from_iterable(paired_cols))

    # Save the inference results (including features and predictions).
    preds_path = output_dir / f"{model.ou_name}_{model.method}_{dataset}_{mode}_preds.csv"
    with preds_path.open("w+") as preds_file:
        temp: NDArray[Any] = np.concatenate((X, y, y_pred), axis=1)  # type: ignore [no-untyped-call]
        test_result_df = pd.DataFrame(
            temp,
            columns=model.features + DIFFED_TARGET_COLS + pred_cols,
        )
        test_result_df[reordered_cols].to_csv(preds_file, float_format="%.1f", index=False)

    # Save the decision tree if relevant.
    if model.method == "dt" and mode == "train":
        # We have one decision tree for each target variable.
        for idx, target_name in enumerate(DIFFED_TARGET_COLS):
            # Generate the dotgraph.
            dot = tree.export_graphviz(
                model.model.estimators_[idx],
                feature_names=model.features,
                filled=True,
            )

            # Construct the file path and save the tree plot.
            dt_file = str(output_dir / f"treeplot_{model.ou_name}_{target_name}.png")
            graphviz.graph_from_dot_data(dot).write_png(dt_file)

    # Evaluate the model performance and write the results to disk.
    ou_eval_path = output_dir / f"summary_{mode}_{dataset}.txt"
    with ou_eval_path.open("w+") as f:
        f.write(
            f"\n============= {mode.title()}: Model Summary for {model.ou_name} Model: {model.method} =============\n"
        )
        f.write(f"Features used: {model.features}\n")
        f.write(f"Num Features used: {len(model.features)}\n")

        # Evaluate performance for every resource consumption metric.
        for target_idx, target in enumerate(DIFFED_TARGET_COLS):
            f.write(f"===== Target: {target} =====\n")
            target_pred = y_pred[:, target_idx]
            target_true = y[:, target_idx]
            true_mean = target_true.mean()
            pred_mean = target_pred.mean()
            mse = mean_squared_error(target_true, target_pred)
            mae = mean_absolute_error(target_true, target_pred)
            mape = mean_absolute_percentage_error(target_true, target_pred)
            rsquared = r2_score(target_true, target_pred)
            f.write(f"Target Mean: {round(true_mean, 2)}, Predicted Mean: {round(pred_mean, 2)}\n")
            f.write(f"Mean Absolute Percentage Error (MAPE): {round(mape, 2)}\n")
            f.write(f"Mean Squared Error (MSE): {round(mse, 2)}\n")
            f.write(f"Mean Absolute Error (MAE): {round(mae, 2)}\n")
            f.write(f"Percentage Explained Variation (R-squared): {round(rsquared, 2)}\n")

        f.write("======================== END SUMMARY ========================\n")


def load_data(data_dir):
    """Load the training data.

    Parameters
    ----------
    data_dir : Path
        Directory from which to load data.

    Returns
    -------
    dict[str, DataFrame]
        A map from operating unit names to their training data.

    Raises
    ------
    Exception
        If there is no valid training data.
    """
    # Load all the OU data from disk given the data directory.
    # We filter all files with zero results because it is common to only have data for
    # a few operating units.
    result_paths: list[Path] = [fp for fp in data_dir.glob("*.csv") if os.stat(fp).st_size > 0]
    ou_name_to_df: dict[str, DataFrame] = {}

    for ou_name in PLAN_NODE_NAMES:
        ou_results = [fp for fp in result_paths if fp.name.startswith(ou_name)]
        if len(ou_results) > 0:
            logger.debug("Found %s run(s) for %s", len(ou_results), ou_name)
            ou_name_to_df[ou_name] = pd.concat(map(pd.read_csv, ou_results))

    # We should always have data for at least one operating unit.
    if len(ou_name_to_df) == 0:
        raise Exception(f"No data found in data_dir: {data_dir}")

    return ou_name_to_df


def prep_train_data(df):
    """Pre-process the training data.

    Parameters
    ----------
    df : DataFrame
        Training data for one operating unit.

    Returns
    -------
    DataFrame
        Pre-processed training data.
    """

    # We must filter metadata columns from the operating unit datasets.
    # Remove all the "undifferenced" target columns, as we only predict
    # the "differenced" resource costs.
    cols_to_remove: list[str] = [
        "rid",
        "statement_id",
        "invocation_id",
        "start_time",
        "end_time",
        "cpu_id",
        "query_id",
        "plan_node_id",
        "pid",
        "statement_timestamp",
        "left_child_plan_node_id",
        "right_child_plan_node_id",
    ] + BASE_TARGET_COLS

    # Remove all columns which include the relation ID.
    relid_cols = [col for col in df.columns if col.endswith("relid")]
    cols_to_remove += relid_cols

    # Don't try to remove any columns which aren't actually in the DataFrame.
    cols_to_remove = [col for col in cols_to_remove if col in df.columns]
    df.drop(cols_to_remove, axis=1, inplace=True)

    # Remove all features with zero variance.
    zero_var_cols = []
    for col in df.columns:
        if df[col].nunique() == 1 and col not in DIFFED_TARGET_COLS:
            zero_var_cols.append(col)

    if zero_var_cols:
        logger.debug("Dropping zero-variance features: %s", zero_var_cols)
        df.drop(zero_var_cols, axis=1, inplace=True)

    # Sort the DataFrame by column for uniform downstream outputs.
    df.sort_index(axis=1, inplace=True)

    # Now that we've filtered the columns, any column that isn't a target must be a feature.
    feat_cols: list[str] = [col for col in df.columns if col not in DIFFED_TARGET_COLS]

    # If there are no remaining features, we still want to make a trivial model.  To do this,
    # we simply insert a single "bias" feature column of 1's.
    if not feat_cols:
        logger.warning("All features were constant.  Defaulting to a single constant bias column.")
        df["bias"] = 1
        feat_cols = ["bias"]

    logger.info("Training Data | Num Observations: %s", df.shape[0])
    logger.info("Num Features: %s | Feature List: %s", len(feat_cols), feat_cols)

    return df


def main(config_file, dir_data_train, dir_data_eval, dir_output):
    # Load modeling configuration.
    if not config_file.exists():
        raise ValueError(f"Config file: {config_file} does not exist")

    with config_file.open("r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)["modeling"]

    # Mark this training-evaluation run with a timestamp for identification.
    training_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Identify which database benchmark(s) to use for training and evaluation.
    train_bench_db = config["train_bench_db"]
    eval_bench_db = config["eval_bench_db"]

    for bench_db in [train_bench_db, eval_bench_db]:
        if bench_db not in BENCHDB_TO_TABLES:
            raise ValueError(f"Benchmark DB {bench_db} not supported")

    # Default to the latest experiment if none is provided.
    if config["experiment_name"] is None:
        experiment_list = sorted([exp_path.name for exp_path in dir_data_train.glob("*")])
        logger.info("%s experiments: %s", train_bench_db, experiment_list)
        assert len(experiment_list) > 0, "No experiments found"
        experiment_name = experiment_list[-1]
        logger.info("Experiment name was not provided, using experiment: %s", experiment_name)
    else:
        experiment_name = config["experiment_name"]

    train_exp_root = dir_data_train / experiment_name
    all_train_names = sorted(
        [d.name for d in train_exp_root.iterdir() if d.is_dir() and d.name.startswith(train_bench_db)]
    )

    eval_exp_root = dir_data_eval / experiment_name
    all_eval_names = sorted(
        [d.name for d in eval_exp_root.iterdir() if d.is_dir() and d.name.startswith(eval_bench_db)]
    )

    # Verify that the training and evaluation data directories exist.
    assert (
        len(all_train_names) > 0 and len(all_eval_names) > 0
    ), f"Benchmark data not found for experiment: {experiment_name}\nMake sure you generated the full sets of data."
    assert len(all_train_names) == len(
        all_eval_names
    ), f"Train/Eval cases mismatch for experiment: {experiment_name}\nMake sure you generated the full sets of data."

    for train_bench_name, eval_bench_name in zip(all_train_names, all_eval_names):
        training_data_dir = train_exp_root / train_bench_name
        eval_data_dir = eval_exp_root / eval_bench_name

        # Load the data and name the model.
        train_ou_to_df = load_data(training_data_dir)
        eval_ou_to_df = load_data(eval_data_dir)
        base_model_name = f"{config_file.stem}_{training_timestamp}_{train_bench_name}"
        output_dir = dir_output / base_model_name

        for ou_name, train_df in train_ou_to_df.items():
            logger.info("Begin Training OU: %s", ou_name)
            df_train = prep_train_data(train_df)

            # Partition the features and targets.
            feat_cols = [col for col in df_train.columns if col not in DIFFED_TARGET_COLS]
            x_train = df_train[feat_cols].values
            y_train = df_train[DIFFED_TARGET_COLS].values

            # Check if no valid training data was found (for the current operating unit).
            if x_train.shape[1] == 0 or y_train.shape[1] == 0:
                logger.warning(
                    "OU: %s has no valid training data, skipping. Feature cols: %s, X_train shape: %s, y_train shape: %s",
                    ou_name,
                    feat_cols,
                    x_train.shape,
                    y_train.shape,
                )
                continue

            # Train one model for each method specified in the modeling configuration.
            for method in config["methods"]:
                logger.info("Training OU: %s with model: %s", ou_name, method)
                ou_model = BehaviorModel(
                    method,
                    ou_name,
                    base_model_name,
                    config,
                    feat_cols,
                )

                # Train the model.
                ou_model.train(x_train, y_train)

                # Save and evaluate the model against the training data.
                full_outdir = output_dir / method / ou_name
                full_outdir.mkdir(parents=True, exist_ok=True)
                ou_model.save(dir_output)
                evaluate(ou_model, train_df, full_outdir, train_bench_name, mode="train")

                if ou_name not in eval_ou_to_df:
                    logger.warning("OU: %s has training data but no evaluation data.", ou_name)
                else:
                    # Evaluate the model against the evaluation data.
                    df_eval = eval_ou_to_df[ou_name]

                    if "bias" in feat_cols:
                        df_eval["bias"] = 1

                    evaluate(ou_model, df_eval, full_outdir, eval_bench_name, mode="eval")


class TrainCLI(cli.Application):
    config_file = cli.SwitchAttr(
        "--config-file",
        Path,
        mandatory=True,
        help="Path to configuration YAML containing modeling parameters.",
    )
    dir_data_train = cli.SwitchAttr(
        "--dir-data-train",
        Path,
        mandatory=True,
        help="Folder containing training data.",
    )
    dir_data_eval = cli.SwitchAttr(
        "--dir-data-eval",
        Path,
        mandatory=True,
        help="Folder containing evaluation data.",
    )
    dir_output = cli.SwitchAttr(
        "--dir-output",
        Path,
        mandatory=True,
        help="Folder to output models to.",
    )

    def main(self):
        main(self.config_file, self.dir_data_train, self.dir_data_eval, self.dir_output)


if __name__ == "__main__":
    TrainCLI.run()
