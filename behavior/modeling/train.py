import itertools
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydotplus
import yaml
from numpy.typing import NDArray
from pandas import DataFrame
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
    CONFIG_DIR,
    EVAL_DATA_DIR,
    LEAF_NODES,
    MODEL_DATA_DIR,
    PLAN_NODE_NAMES,
    TRAIN_DATA_DIR,
    get_logger,
)
from behavior.modeling.model import BehaviorModel


def evaluate(
    ou_model: BehaviorModel,
    X: NDArray[np.float32],
    y: NDArray[np.float32],
    output_dir: Path,
    dataset: str,
    mode: str,
) -> None:
    if mode not in ["train", "eval"]:
        raise ValueError(f"Invalid mode: {mode}")

    y_pred = ou_model.predict(X)

    # pair and reorder the target columns for readable outputs
    paired_cols = zip([f"pred_{col}" for col in ou_model.targets], ou_model.targets)
    reordered_cols = ou_model.features + list(
        itertools.chain.from_iterable(paired_cols)
    )

    preds_path = (
        output_dir / f"{ou_model.ou_name}_{ou_model.method}_{dataset}_{mode}_preds.csv"
    )
    with preds_path.open("w+") as preds_file:
        temp: NDArray[Any] = np.concatenate((X, y, y_pred), axis=1)  # type: ignore [no-untyped-call]
        test_result_df = pd.DataFrame(
            temp,
            columns=ou_model.features
            + ou_model.targets
            + [f"pred_{col}" for col in ou_model.targets],
        )
        test_result_df[reordered_cols].to_csv(
            preds_file, float_format="%.1f", index=False
        )

    if ou_model.method == "dt" and mode == "train":
        for idx, target_name in enumerate(ou_model.targets):
            dot = tree.export_graphviz(
                ou_model.model.estimators_[idx],
                feature_names=ou_model.features,
                filled=True,
            )
            dt_file = (
                f"{output_dir}/{ou_model.ou_name}_{mode}_treeplot_{target_name}.png"
            )
            pydotplus.graphviz.graph_from_dot_data(dot).write_png(dt_file)

    ou_eval_path = (
        output_dir
        / f"{ou_model.ou_name}_{ou_model.method}_{dataset}_{mode}_summary.txt"
    )
    with ou_eval_path.open("w+") as eval_file:
        eval_file.write(
            f"\n============= {mode.title()}: Model Summary for {ou_model.ou_name} Model: {ou_model.method} =============\n"
        )
        eval_file.write(f"Features used: {ou_model.features}\n")
        eval_file.write(f"Num Features used: {len(ou_model.features)}\n")
        eval_file.write(f"Targets estimated: {ou_model.targets}\n")

        for target_idx, target in enumerate(ou_model.targets):
            eval_file.write(f"===== Target: {target} =====\n")
            target_pred = y_pred[:, target_idx]
            target_true = y[:, target_idx]
            true_mean = target_true.mean()
            pred_mean = target_pred.mean()
            mse = mean_squared_error(target_true, target_pred)
            mae = mean_absolute_error(target_true, target_pred)
            mape = mean_absolute_percentage_error(target_true, target_pred)
            rsquared = r2_score(target_true, target_pred)
            eval_file.write(
                f"Target Mean: {round(true_mean, 2)}, Predicted Mean: {round(pred_mean, 2)}\n"
            )
            eval_file.write(
                f"Mean Absolute Percentage Error (MAPE): {round(mape, 2)}\n"
            )
            eval_file.write(f"Mean Squared Error (MSE): {round(mse, 2)}\n")
            eval_file.write(f"Mean Absolute Error (MAE): {round(mae, 2)}\n")
            eval_file.write(
                f"Percentage Explained Variation (R-squared): {round(rsquared, 2)}\n"
            )

        eval_file.write(
            "======================== END SUMMARY ========================\n"
        )


def load_data(data_dir: Path) -> dict[str, DataFrame]:
    result_paths: list[Path] = [
        fp for fp in data_dir.glob("*.csv") if os.stat(fp).st_size > 0
    ]
    ou_name_to_df: dict[str, DataFrame] = {}

    for ou_name in PLAN_NODE_NAMES:
        ou_results = [fp for fp in result_paths if fp.name.startswith(ou_name)]
        if len(ou_results) > 0:
            get_logger().info("Found %s run(s) for %s", len(ou_results), ou_name)
            ou_name_to_df[ou_name] = pd.concat(map(pd.read_csv, ou_results))

    if len(ou_name_to_df) == 0:
        raise Exception(f"No data found in data_dir: {data_dir}")

    return ou_name_to_df


def prep_train_data(
    ou_name: str, df: DataFrame, feat_diff: bool, target_diff: bool
) -> tuple[list[str], list[str], NDArray[Any], NDArray[Any]]:
    cols_to_remove: list[str] = [
        "start_time",
        "end_time",
        "cpu_id",
        "query_id",
        "rid",
        "plan_node_id",
        "statement_timestamp",
        "left_child_plan_node_id",
        "right_child_plan_node_id",
    ]

    diff_targ_cols = [f"diffed_{col}" for col in BASE_TARGET_COLS]

    if target_diff and ou_name not in LEAF_NODES:
        cols_to_remove += BASE_TARGET_COLS
    else:
        cols_to_remove += diff_targ_cols

    cols_to_remove = [col for col in cols_to_remove if col in df.columns]

    for col in df.columns:
        if df[col].nunique() == 1:
            cols_to_remove.append(col)

    df = df.drop(cols_to_remove, axis=1).sort_index(axis=1)

    if len(cols_to_remove) > 0:
        logger = get_logger()
        logger.info("Dropped zero-variance columns: %s", cols_to_remove)
        logger.info(
            "Num Remaining: %s, Num Removed: %s", len(df.columns), len(cols_to_remove)
        )

    if target_diff and ou_name not in LEAF_NODES:
        print(f"using differenced targets for: {ou_name}")
        target_cols = [col for col in df.columns if col in diff_targ_cols]
    else:
        target_cols = [col for col in df.columns if col in BASE_TARGET_COLS]
        print(f"using undiff targets for: {ou_name}.  target_cols: {target_cols}")
    all_target_cols = BASE_TARGET_COLS + diff_targ_cols
    feat_cols: list[str] = [col for col in df.columns if col not in all_target_cols]

    if not feat_diff:
        feat_cols = [col for col in feat_cols if not col.startswith("diffed")]

    print(f"Using features: {feat_cols}")

    X = df[feat_cols].values
    y = df[target_cols].values

    return feat_cols, target_cols, X, y


def prep_eval_data(
    df: pd.DataFrame, feat_cols: list[str], target_cols: list[str]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    X = df[feat_cols].values
    y = df[target_cols].values

    return X, y


def main(config_name: str) -> None:
    # load config
    config_path: Path = CONFIG_DIR / f"{config_name}.yaml"
    if not config_path.exists():
        raise ValueError(f"Config file: {config_name} does not exist")

    with config_path.open("r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)["modeling"]

    training_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_bench_dbs = config["train_bench_dbs"]
    train_bench_db = train_bench_dbs[0]
    eval_bench_dbs = config["eval_bench_dbs"]
    eval_bench_db = eval_bench_dbs[0]
    feat_diff = config["features_diff"]
    target_diff = config["targets_diff"]
    logger = get_logger()

    for train_bench_db in train_bench_dbs:
        if train_bench_db not in BENCHDB_TO_TABLES:
            raise ValueError(f"Benchmark DB {config['bench_db']} not supported")

    # if no experiment name is provided, try to find one
    if config["experiment_name"] is None:
        experiment_list = sorted(
            [exp_path.name for exp_path in TRAIN_DATA_DIR.glob("*")]
        )
        logger.warning("%s experiments: %s", train_bench_db, experiment_list)
        assert len(experiment_list) > 0, "No experiments found"
        experiment_name = experiment_list[-1]
        logger.warning(
            "Experiment name was not provided, using experiment: %s", experiment_name
        )

    training_data_dir = (
        TRAIN_DATA_DIR / experiment_name / train_bench_db / "differenced"
    )
    eval_data_dir = EVAL_DATA_DIR / experiment_name / eval_bench_db / "differenced"

    logger.warning("eval data dir: %s", eval_data_dir)
    if not training_data_dir.exists():
        raise ValueError(
            f"Train Benchmark DB {train_bench_db} not found in experiment: {experiment_name}"
        )
    if not eval_data_dir.exists():
        raise ValueError(
            f"Eval Benchmark DB {eval_bench_db} not found in experiment: {experiment_name}"
        )

    train_ou_to_df = load_data(training_data_dir)
    eval_ou_to_df = load_data(eval_data_dir)
    base_model_name = f"{config_name}_{training_timestamp}"
    output_dir = MODEL_DATA_DIR / base_model_name

    for ou_name, train_df in train_ou_to_df.items():
        logger.warning("Begin Training OU: %s", ou_name)
        feat_cols, target_cols, x_train, y_train = prep_train_data(
            ou_name, train_df, feat_diff, target_diff
        )

        if x_train.shape[1] == 0 or y_train.shape[1] == 0:
            print(feat_cols)
            print(target_cols)
            print(x_train.shape)
            print(y_train.shape)
            logger.warning("%s has no valid training data, skipping", ou_name)
            continue

        for method in config["methods"]:
            full_outdir = output_dir / method / ou_name
            full_outdir.mkdir(parents=True, exist_ok=True)
            logger.warning("Training OU: %s with model: %s", ou_name, method)
            ou_model = BehaviorModel(
                method, ou_name, base_model_name, config, feat_cols, target_cols
            )
            ou_model.train(x_train, y_train)
            ou_model.save()
            evaluate(
                ou_model, x_train, y_train, full_outdir, train_bench_db, mode="train"
            )

            if ou_name in eval_ou_to_df:
                x_eval, y_eval = prep_eval_data(
                    eval_ou_to_df[ou_name], feat_cols, target_cols
                )
                evaluate(
                    ou_model, x_eval, y_eval, full_outdir, eval_bench_db, mode="eval"
                )
