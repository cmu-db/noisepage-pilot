import pandas as pd
from doit.action import CmdAction

import dodos.noisepage
from dodos import VERBOSITY_DEFAULT, default_artifacts_path, default_build_path

ARTIFACTS_PATH = default_artifacts_path()
BUILD_PATH = default_build_path()

# Input: query log.
QUERY_LOG_DIR = dodos.noisepage.ARTIFACT_pgdata_log
from pathlib import Path
QUERY_LOG_DIR = Path("/home/mkpjnx/repos/noisepage-pilot/forecast/data/extracted/long_simple")

# Scratch work.
PREPROCESSOR_ARTIFACT = BUILD_PATH / "preprocessed.parquet.gzip"
PREPROCESSOR_TIMESTAMP = BUILD_PATH / "preprocessed.timestamp.txt"
CLUSTER_ARTIFACT = BUILD_PATH / "clustered.parquet"
MODEL_DIR = BUILD_PATH / "models"

# Default Forecasting Params
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

    def forecast_action(pred_start, pred_end, pred_horizon, pred_interval, pred_seqlen):
        log_start = None
        log_end = None
        with open(PREPROCESSOR_TIMESTAMP) as ts_file:
            lines = ts_file.readlines()
            assert len(lines) >= 2, "Timestamp file should have two lines with a timestamp each"
            log_start = pd.Timestamp(lines[0]).floor(pred_interval)
            log_end = pd.Timestamp(lines[1]).floor(pred_interval)

        # The minimum prediction horizon needed
        if pred_start is None:
            pred_start = log_end + pred_interval
        if pred_end is None:
            pred_end = log_end + pred_horizon

        # TODO(Mike): assert there is enough data for inference
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
                "help": "The start point of the forecast (inclusive). Default: now.",
                "type": pd.Timestamp,
                "default": None,
            },
            {
                "name": "pred_end",
                "long": "pred_end",
                "help": "The end point of the forecast (inclusive). Default: 1 minute from now.",
                "type": pd.Timestamp,
                "default": None,
            },
            {
                "name": "pred_horizon",
                "long": "pred_horizon",
                "help": "The end point of the forecast (inclusive). Default: 1 minute from now.",
                "type": pd.Timedelta,
                "default": DEFAULT_PRED_HORIZON,  # Infer horizon from file if needed
            },
            {
                "name": "pred_interval",
                "long": "pred_interval",
                "help": "The end point of the forecast (inclusive). Default: 1 minute from now.",
                "type": pd.Timedelta,
                "default": DEFAULT_PRED_INTERVAL,
            },
            {
                "name": "pred_seqlen",
                "long": "pred_seqlen",
                "help": "The end point of the forecast (inclusive). Default: 1 minute from now.",
                "type": int,
                "default": DEFAULT_PRED_SEQLEN,
            },
        ],
    }
