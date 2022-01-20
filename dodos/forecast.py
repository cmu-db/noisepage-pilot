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
CLUSTER_ARTIFACT = BUILD_PATH / "clustered.parquet"
MODEL_DIR = BUILD_PATH / "models"

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

    def forecast_action(time_start, time_end):
        return (
            "python3 ./forecast/forecaster.py "
            f"--preprocessor-parquet {PREPROCESSOR_ARTIFACT} "
            f"--clusterer-parquet {CLUSTER_ARTIFACT} "
            f"--model-path {MODEL_DIR} "
            f'--start-time "{time_start}" '
            f'--end-time "{time_end}" '
            f"--output-csv {ARTIFACT_FORECAST} "
            "--override-models "  # TODO(Mike): Always override models?
        )

    return {
        "actions": [
            f"mkdir -p {MODEL_DIR}",
            CmdAction(forecast_action),
        ],
        "file_dep": ["./forecast/forecaster.py", PREPROCESSOR_ARTIFACT, CLUSTER_ARTIFACT],
        "targets": [ARTIFACT_FORECAST],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "time_start",
                "long": "time_start",
                "help": "The start point of the forecast (inclusive). Default: now.",
                "default": pd.Timestamp.now().tz_localize("EST"),
            },
            {
                "name": "time_end",
                "long": "time_end",
                "help": "The end point of the forecast (inclusive). Default: 1 minute from now.",
                "default": pd.Timestamp.now().tz_localize("EST") + pd.Timedelta(seconds=60),
            },
        ],
    }
