import pandas as pd
from doit.action import CmdAction

import dodos.noisepage
from dodos import VERBOSITY_DEFAULT, default_artifacts_path, default_build_path

ARTIFACTS_PATH = default_artifacts_path()
BUILD_PATH = default_build_path()

# Input: query log.
QUERY_LOG_DIR = dodos.noisepage.ARTIFACT_pgdata_log

# Scratch work.
PREPROCESSOR_ARTIFACT = BUILD_PATH / "preprocessed.parquet.gzip"
PREPROCESSOR_TIMESTAMP = BUILD_PATH / "preprocessed.timestamp.txt"
CLUSTER_ARTIFACT = BUILD_PATH / "clustered.parquet"
MODEL_DIR = BUILD_PATH / "models"

# Default forecasting parameters.
DEFAULT_PRED_HORIZON = pd.Timedelta(seconds=10)
DEFAULT_PRED_INTERVAL = pd.Timedelta(seconds=1)
DEFAULT_PRED_SEQLEN = 10

# Output: predictions.
ARTIFACT_FORECAST = ARTIFACTS_PATH / "forecast.csv"


def task_forecast_preprocess():
    """
    Forecast: preprocess the query logs by extracting query templates.
    """

    def preprocessor_action():
        return (
            "python3 ./forecast/preprocessor.py "
            f"--query-log-folder {QUERY_LOG_DIR} "
            f"--output-parquet {PREPROCESSOR_ARTIFACT} "
            f"--output-timestamp {PREPROCESSOR_TIMESTAMP} "
        )

    return {
        "actions": [
            f"mkdir -p {ARTIFACTS_PATH}",
            f"mkdir -p {BUILD_PATH}",
            # Preprocess the PostgreSQL query logs.
            CmdAction(preprocessor_action),
        ],
        "file_dep": ["./forecast/preprocessor.py", *QUERY_LOG_DIR.glob("*")],
        "targets": [PREPROCESSOR_ARTIFACT],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_forecast_cluster():
    """
    Forecast: cluster the preprocessed queries.
    """

    def cluster_action():
        return (
            "python3 ./forecast/clusterer.py "
            f"--preprocessor-parquet {PREPROCESSOR_ARTIFACT} "
            f"--output-parquet {CLUSTER_ARTIFACT} "
        )

    return {
        "actions": [CmdAction(cluster_action)],
        "file_dep": ["./forecast/clusterer.py", PREPROCESSOR_ARTIFACT],
        "targets": [CLUSTER_ARTIFACT],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_forecast_predict():
    """
    Forecast: produce predictions for the given time range.
    """

    # Read the query log timestamps from the preprocessor's output.
    with open(PREPROCESSOR_TIMESTAMP) as ts_file:
        lines = ts_file.readlines()
        assert len(lines) == 2, "Timestamp file should have two lines with a timestamp each."
        log_start = pd.Timestamp(lines[0])
        log_end = pd.Timestamp(lines[1])

    def forecast_action(pred_start, pred_end, pred_horizon, pred_interval, pred_seqlen):
        nonlocal log_start, log_end
        log_start = log_start.floor(pred_interval)
        log_end = log_end.floor(pred_interval)

        # Infer the prediction window from the query log and default parameters.
        if pred_start is None:
            pred_start = log_end + pred_interval
        if pred_end is None:
            pred_end = log_end + pred_horizon

        # TODO(Mike): Assert there is enough data for inference.
        # TODO(WAN): This entire callable may be invoked repeatedly, so print statements are not a good idea.
        #   Arguably we can push this logic into the forecaster itself (a verbose mode). I am ok with this for now,
        #   but if you the reader are thinking of duplicating this code, please don't.
        print(
            f"Using query data ({log_start.isoformat()} to {log_end.isoformat()})\n"
            f"to predict ({pred_start.isoformat()} to {pred_end.isoformat()})\n"
            f"with horizon {pred_horizon}, interval {pred_interval}, seqlen {pred_seqlen}"
        )

        return (
            "python3 ./forecast/forecaster.py "
            f"--preprocessor-parquet {PREPROCESSOR_ARTIFACT} "
            f"--clusterer-parquet {CLUSTER_ARTIFACT} "
            f"--model-path {MODEL_DIR} "
            f'--start-time "{pred_start}" '
            f'--end-time "{pred_end}" '
            f"--output-csv {ARTIFACT_FORECAST} "
            f"--horizon {pred_horizon.isoformat()} "
            f"--interval {pred_interval.isoformat()} "
            f"--seqlen {pred_seqlen} "
            "--override-models "  # TODO(Mike): Always override models?
        )

    return {
        "actions": [
            f"mkdir -p {MODEL_DIR}",
            CmdAction(forecast_action),
        ],
        "file_dep": ["./forecast/forecaster.py", PREPROCESSOR_ARTIFACT, PREPROCESSOR_TIMESTAMP, CLUSTER_ARTIFACT],
        "targets": [ARTIFACT_FORECAST],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "pred_start",
                "long": "pred_start",
                "help": "The start point of the forecast (inclusive). Default: last timestamp in the query log.",
                "type": pd.Timestamp,
                "default": None,
            },
            {
                "name": "pred_end",
                "long": "pred_end",
                "help": "The end point of the forecast (inclusive). Default: 10 seconds from pred_start.",
                "type": pd.Timestamp,
                "default": None,
            },
            {
                "name": "pred_horizon",
                "long": "pred_horizon",
                "help": "How far in the future to predict.",
                "type": pd.Timedelta,
                "default": DEFAULT_PRED_HORIZON,  # Infer horizon from file if needed
            },
            {
                "name": "pred_interval",
                "long": "pred_interval",
                "help": "Interval to aggregate the queries for training/prediction.",
                "type": pd.Timedelta,
                "default": DEFAULT_PRED_INTERVAL,
            },
            {
                "name": "pred_seqlen",
                "long": "pred_seqlen",
                "help": "How many consecutive intervals of query arrival rate is used to make inference.",
                "type": int,
                "default": DEFAULT_PRED_SEQLEN,
            },
        ],
    }
