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

from behavior import BASE_TARGET_COLS, PLAN_NODE_NAMES
from behavior.modeling import BLOCKED_FEATURES, featurize
from behavior.modeling.model import BehaviorModel

logger = logging.getLogger(__name__)


def evaluate(model, df, output_dir, mode):
    """Evaluate the model.

    Parameters
    ----------
    ou_model : BehaviorModel
        Model to evaluate.
    df : DataFrame
        Evaluation data.
    output_dir : Path
        Results output directory.
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
    X = featurize.extract_input_features(df, model.features)
    y = df[BASE_TARGET_COLS].values

    # Run inference.
    y_pred = model.predict(X)

    # Pair and re-order the target columns for more readable outputs.
    pred_cols = [f"pred_{col}" for col in BASE_TARGET_COLS]
    paired_cols = zip(pred_cols, BASE_TARGET_COLS)
    reordered_cols = model.features + list(itertools.chain.from_iterable(paired_cols))

    # Save the inference results (including features and predictions).
    preds_path = output_dir / f"{model.ou_name}_{model.method}_{mode}_preds.csv"
    with preds_path.open("w+") as preds_file:
        temp: NDArray[Any] = np.concatenate((X, y, y_pred), axis=1)  # type: ignore [no-untyped-call]
        test_result_df = pd.DataFrame(
            temp,
            columns=model.features + BASE_TARGET_COLS + pred_cols,
        )
        test_result_df[reordered_cols].to_csv(preds_file, float_format="%.1f", index=False)

    # Save the decision tree if relevant.
    if model.method == "dt" and mode == "train":
        # We have one decision tree for each target variable.
        for idx, target_name in enumerate(BASE_TARGET_COLS):
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
    ou_eval_path = output_dir / f"summary_{mode}.txt"
    with ou_eval_path.open("w+") as f:
        f.write(
            f"\n============= {mode.title()}: Model Summary for {model.ou_name} Model: {model.method} =============\n"
        )
        f.write(f"Features used: {model.features}\n")
        f.write(f"Num Features used: {len(model.features)}\n")

        # Evaluate performance for every resource consumption metric.
        for target_idx, target in enumerate(BASE_TARGET_COLS):
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


def load_data(data_dirs):
    """Load the training data.

    Parameters
    ----------
    data_dir : List[Path]
        Directories from which to load data.

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
    result_paths = []
    for data_dir in data_dirs:
        result_paths.extend([fp for fp in Path(data_dir).glob("*.csv") if os.stat(fp).st_size > 0])

    ou_name_to_df: dict[str, DataFrame] = {}
    for ou_name in PLAN_NODE_NAMES:
        ou_results = [fp for fp in result_paths if fp.name == f"{ou_name}.csv"]
        if len(ou_results) > 0:
            logger.debug("Found %s run(s) for %s", len(ou_results), ou_name)
            ou_name_to_df[ou_name] = pd.concat(map(pd.read_csv, ou_results))

    # We should always have data for at least one operating unit.
    if len(ou_name_to_df) == 0:
        raise Exception(f"No data found in data_dirs: {data_dirs}")

    return ou_name_to_df


def prep_input_data(df):
    """Pre-process the input data.

    Parameters
    ----------
    df : DataFrame
        Input (train/test) data for one operating unit.
    """
    # Remove all features that are blocked.
    blocked = [col for col in df.columns for block in BLOCKED_FEATURES if col.endswith(block)]

    # Don't try to remove any columns which aren't actually in the DataFrame.
    blocked = [col for col in blocked if col in df.columns]
    df.drop(blocked, axis=1, inplace=True)

    # Sort the DataFrame by column for uniform downstream outputs.
    df.sort_index(axis=1, inplace=True)


def glob_files(base_dir, experiment_names, benchmark_names, train):
    mode = "train" if train else "eval"
    if experiment_names is None:
        experiment_list = sorted([exp_path.name for exp_path in base_dir.glob("*")])
        assert len(experiment_list) > 0, f"No experiments found {base_dir}"
        experiment_names = [experiment_list[-1]]
    else:
        experiment_list = []
        for pattern in experiment_names:
            experiment_list.extend([exp_path.name for exp_path in base_dir.glob(pattern)])
        assert len(experiment_list) > 0, f"No experiments found {base_dir} with {experiment_names}"
        experiment_names = experiment_list

    glob_results = set()
    for experiment_name in experiment_names:
        exp_root = base_dir / experiment_name / mode
        for benchmark_name in benchmark_names:
            glob_results.update([str(exp_root / path.name) for path in exp_root.glob(benchmark_name)])

    return list(glob_results)


def main(
    config_file,
    dir_data,
    dir_output,
    train_experiment_names,
    train_benchmark_names,
    eval_experiment_names,
    eval_benchmark_names,
    use_featurewiz,
):
    # Load modeling configuration.
    if not config_file.exists():
        raise ValueError(f"Config file: {config_file} does not exist")

    with config_file.open("r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)["modeling"]

    # Mark this training-evaluation run with a timestamp for identification.
    training_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Derive all the relevant directories for training data and evaluation data.
    train_files = glob_files(dir_data, train_experiment_names, train_benchmark_names, True)
    eval_files = glob_files(dir_data, eval_experiment_names, eval_benchmark_names, False)
    assert len(train_files) > 0, "No matching data files for training could be found."
    assert len(eval_files) > 0, "No matching data files for evaluating could be found."

    # Load the data and name the model.
    train_ou_to_df = load_data(train_files)
    eval_ou_to_df = load_data(eval_files)
    base_model_name = f"{config_file.stem}_{training_timestamp}"
    output_dir = dir_output / base_model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "source.txt").open("w+") as f:
        f.write(f"Train: {train_files}\n")
        f.write(f"Eval: {eval_files}\n")

    # Prepare all the input evaluation data.
    _ = [prep_input_data(df) for (_, df) in eval_ou_to_df.items()]

    for ou_name, df_train in train_ou_to_df.items():
        logger.info("Begin Training OU: %s", ou_name)
        prep_input_data(df_train)
        df_eval = eval_ou_to_df[ou_name] if ou_name in eval_ou_to_df else None

        if use_featurewiz:
            # TODO(wz2): Currently prioritize the `elapsed_us` target. We may want to specify multiple
            # targets in the future and/or consider training a model for each target separately.
            features = featurize.derive_input_features(
                df_train, test=df_eval, targets=["elapsed_us"], config=config["featurize"]
            )
        else:
            # Get a metadata representation for extracting all feature columns.
            features = featurize.extract_all_features(df_train)

        # Partition the features and targets.
        x_train = featurize.extract_input_features(df_train, features)
        y_train = df_train[BASE_TARGET_COLS].values

        # Check if no valid training data was found (for the current operating unit).
        if x_train.shape[1] == 0 or y_train.shape[1] == 0:
            logger.warning(
                "OU: %s has no valid training data, skipping. Feature cols: %s, X_train shape: %s, y_train shape: %s",
                ou_name,
                features,
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
                features,
            )

            # Train the model.
            ou_model.train(x_train, y_train)

            # Save and evaluate the model against the training data.
            full_outdir = output_dir / method / ou_name
            full_outdir.mkdir(parents=True, exist_ok=True)
            ou_model.save(dir_output)
            evaluate(ou_model, df_train, full_outdir, mode="train")

            if ou_name not in eval_ou_to_df:
                logger.warning("OU: %s has training data but no evaluation data.", ou_name)
            else:
                # Evaluate the model against the evaluation data.
                evaluate(ou_model, df_eval, full_outdir, mode="eval")


class TrainCLI(cli.Application):
    config_file = cli.SwitchAttr(
        "--config-file",
        Path,
        mandatory=True,
        help="Path to configuration YAML containing modeling parameters.",
    )
    dir_data = cli.SwitchAttr(
        "--dir-data",
        Path,
        mandatory=True,
        help="Folder containing all training and evaluation data.",
    )
    dir_output = cli.SwitchAttr(
        "--dir-output",
        Path,
        mandatory=True,
        help="Folder to output models to.",
    )
    train_experiment_names = cli.SwitchAttr(
        "--train-experiment-names",
        str,
        mandatory=False,
        default=None,
        help="Comma separated list of experiments and/or experiments glob patterns to train models from.",
    )
    train_benchmark_names = cli.SwitchAttr(
        "--train-benchmark-names",
        str,
        mandatory=False,
        default="*",
        help="Comma separated list of benchmarks and/or benchmarks glob patterns to train models from.",
    )
    eval_experiment_names = cli.SwitchAttr(
        "--eval-experiment-names",
        str,
        mandatory=False,
        default=None,
        help="Comma separated list of experiments and/or experiments glob patterns to evaluate models on.",
    )
    eval_benchmark_names = cli.SwitchAttr(
        "--eval-benchmark-names",
        str,
        mandatory=False,
        default="*",
        help="Comma separated list of benchmarks and/or benchmarks glob patterns to evaluate models on.",
    )
    use_featurewiz = cli.Flag(
        "--use-featurewiz",
        help="Whether to use featurewiz for feature selection.",
    )

    def main(self):
        self.train_experiment_names = (
            None if not self.train_experiment_names else self.train_experiment_names.split(",")
        )
        self.train_benchmark_names = None if not self.train_benchmark_names else self.train_benchmark_names.split(",")
        self.eval_experiment_names = None if not self.eval_experiment_names else self.eval_experiment_names.split(",")
        self.eval_benchmark_names = None if not self.eval_benchmark_names else self.eval_benchmark_names.split(",")

        main(
            self.config_file,
            self.dir_data,
            self.dir_output,
            self.train_experiment_names,
            self.train_benchmark_names,
            self.eval_experiment_names,
            self.eval_benchmark_names,
            self.use_featurewiz,
        )


if __name__ == "__main__":
    TrainCLI.run()
