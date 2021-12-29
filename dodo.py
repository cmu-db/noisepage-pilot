import glob
from pathlib import Path

import doit
import os

from doit.action import CmdAction

# doit verbosity default controls what to print.
# 0 = nothing, 1 = stderr only, 2 = stdout and stderr.
VERBOSITY_DEFAULT = 2


def task_openspiel_build():
    """
    Invoke CMake for OpenSpiel.
    """
    source_path = (
        lambda: Path(doit.get_initial_workdir()) / "action/selection/open_spiel/"
    )
    build_path = lambda: Path("build/action/selection/open_spiel/build/")
    return {
        "actions": [
            # Set up directories.
            f"mkdir -p {build_path()}",
            # Invoke CMake.
            lambda: os.chdir(build_path()),
            f"cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ {source_path()}",
            # Reset working directory.
            lambda: os.chdir(doit.get_initial_workdir()),
        ],
        "file_dep": glob.glob(str(source_path() / "*")),
        "targets": [f"{build_path() / 'Makefile'}"],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_openspiel_compile():
    """
    Build OpenSpiel.
    """
    build_path = lambda: Path("build/action/selection/open_spiel/build/")
    return {
        "actions": [
            "mkdir -p artifacts",
            # Build database_game.
            lambda: os.chdir(build_path()),
            "make -j database_game",
            # Copy the built binary out.
            lambda: os.chdir(doit.get_initial_workdir()),
            f"cp {build_path() / 'database_game'} ./artifacts/database_game",
            # Reset working directory.
            lambda: os.chdir(doit.get_initial_workdir()),
        ],
        "file_dep": [f"{build_path() / 'Makefile'}"],
        "targets": [f"./artifacts/database_game"],
        "uptodate": [False],  # Always try to recompile.
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_forecast():
    """
    Forecast the workload.
    """
    return {
        "actions": [
            "mkdir -p artifacts",
            # Preprocess the PostgreSQL query logs.
            "python3 ./forecast/preprocessor.py --query-log-folder ./forecast/data/extracted/extended/ --output-hdf ./artifacts/preprocessor.hdf",
            # Cluster the processed query logs.
            # TODO(WAN): clusterer shouldn't go directly to the predictions, plug in Mike's part.
            "python3 ./forecast/clusterer.py --preprocessor-hdf ./artifacts/preprocessor.hdf --output-csv ./artifacts/forecast.csv",
        ],
        "targets": ["./artifacts/preprocessor.hdf", "./artifacts/forecast.csv"],
        "uptodate": [False],  # TODO(WAN): Always recompute?
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_action_generation():
    """
    Generate actions to choose from.
    """
    return {
        "actions": [
            "mkdir -p artifacts",
            # Generate create index suggestions for TPC-C.
            f"python3 ./action/generation/generate_create_index_tpcc.py --min-num-cols 1 --max-num-cols 4 --output-sql ./artifacts/actions.sql",
        ],
        "targets": ["./artifacts/actions.sql"],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_action_recommendation():
    """
    Apply recommended actions to the DBMS.
    """

    def index_picker(db_conn_string):
        action = (
            f"python3 action/recommendation/index_picker.py "
            # index_picker.py arguments.
            "--database-game-path ./artifacts/database_game "
            "-- "
            # database_game arguments.
            f'--db_conn_string "{db_conn_string}" '
            "--actions_path ./artifacts/actions.sql "
            "--forecast_path ./artifacts/forecast.csv"
        )
        return action

    return {
        "actions": [CmdAction(index_picker)],
        "file_dep": [
            "./artifacts/database_game",
            "./artifacts/actions.sql",
            "./artifacts/forecast.csv",
        ],
        "verbosity": VERBOSITY_DEFAULT,
        "uptodate": [False],
        "params": [
            {
                "name": "db_conn_string",
                "long": "--db_conn_string",
                "help": "The database connection string to use.",
                "default": "host=127.0.0.1 port=5432 dbname=spiel user=spiel password=spiel sslmode=disable application_name=psql",
            },
        ],
    }


def task_ci_python():
    """
    This should be run and all warnings fixed before pushing commits.
    """
    folders = ["action", "forecast"]

    return {
        "actions": [
            "black dodo.py",
            *[f"isort {folder}" for folder in folders],
            *[f"black --verbose {folder}" for folder in folders],
            *[f"flake8 {folder}" for folder in folders],
        ],
        "verbosity": VERBOSITY_DEFAULT,
        "uptodate": [False],
    }
