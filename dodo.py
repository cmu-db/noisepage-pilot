import glob
from pathlib import Path

import doit
import os

from doit.action import CmdAction

# doit verbosity default controls what to print.
# 0 = nothing, 1 = stderr only, 2 = stdout and stderr.
VERBOSITY_DEFAULT = 2

DB_CONN_STRING_PILOT_DAEMON = "host=127.0.0.1 port=5432 dbname=np_pilot user=np_pilot_daemon password=np_pilot_daemon_pass sslmode=disable application_name=psql"
DB_CONN_STRING_PILOT_CLIENT = "host=127.0.0.1 port=5432 dbname=np_pilot user=np_pilot_client password=np_pilot_client_pass sslmode=disable application_name=psql"
DB_CONN_STRING_AS_SPIEL = "host=127.0.0.1 port=5432 dbname=np_as_spiel user=np_as_spiel_user password=np_as_spiel_user_pass sslmode=disable application_name=psql"


def task_bootstrap():
    """
    All components: set up the databases and users.
    """
    return {
        "actions": [
            "mkdir -p build",
            "echo 'Bootstrapping required databases as postgres user. You may see a sudo prompt.'",
            f"sudo --user postgres psql --file ./bootstrap.sql > build/bootstrap.out 2>&1",
            f"sudo --reset-timestamp",
            "echo 'Bootstrap complete; sudo permissions reset.'",
        ],
        "file_dep": ["./bootstrap.sql"],
        "targets": ["./build/bootstrap.out"],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_openspiel_build():
    """
    Action selection: invoke CMake for OpenSpiel.
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
    Action selection: build OpenSpiel.
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
    Forecast: forecast the workload.
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
    Action generation: generate actions to choose from.
    """
    return {
        "actions": [
            "mkdir -p artifacts",
            # Generate create index suggestions for TPC-C.
            f"python3 ./action/generation/generate_create_index_tpcc.py --min-num-cols 1 --max-num-cols 4 --output-sql ./artifacts/actions.sql",
        ],
        "file_dep": ["./action/generation/generate_create_index_tpcc.py"],
        "targets": ["./artifacts/actions.sql"],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_action_recommendation():
    """
    Action recommendation: apply recommended actions to the DBMS.
    """

    def index_picker(batch_size, db_conn_string):
        action = (
            f"python3 action/recommendation/index_picker.py "
            # index_picker.py arguments.
            "--database-game-path ./artifacts/database_game "
            f"--batch-size {batch_size} "
            "-- "
            # database_game arguments.
            f'--db_conn_string "{db_conn_string}" '
            "--actions_path ./artifacts/actions.sql "
            "--forecast_path ./artifacts/forecast.csv "
        )
        return action

    return {
        "actions": [CmdAction(index_picker)],
        "file_dep": [
            "./build/bootstrap.out",
            "./artifacts/database_game",
            "./artifacts/actions.sql",
            "./artifacts/forecast.csv",
        ],
        "verbosity": VERBOSITY_DEFAULT,
        "uptodate": [False],
        "params": [
            # index_picker parameters.
            {
                "name": "batch_size",
                "long": "batch_size",
                "help": "The batch size to use for actions.",
                "default": 2500,
            },
            # database_game parameters.
            {
                "name": "db_conn_string",
                "long": "db_conn_string",
                "help": "The database connection string to use.",
                "default": DB_CONN_STRING_AS_SPIEL,
            },
        ],
    }


def task_pilot_bootstrap():
    """
    Pilot: bootstrap the Pilot tables.
    """

    def bootstrap(db_conn_string):
        action = f'psql "{db_conn_string}" --file ./pilot/bootstrap_tables.sql > ./build/bootstrap_pilot.out 2>&1'
        return action

    return {
        "actions": [
            "mkdir -p build",
            CmdAction(bootstrap),
        ],
        "verbosity": VERBOSITY_DEFAULT,
        "file_dep": ["./build/bootstrap.out", "./pilot/bootstrap_tables.sql"],
        "targets": ["./build/bootstrap_pilot.out"],
        "params": [
            {
                "name": "db_conn_string",
                "long": "db_conn_string",
                "help": "Connection string to Pilot database as server.",
                "default": DB_CONN_STRING_PILOT_DAEMON,
            },
        ],
    }


def task_pilot_daemon():
    """
    Pilot: run the Pilot daemon.
    """

    def run_daemon(db_conn_string):
        action = (
            f"python3 pilot/daemon.py "
            # daemon.py arguments.
            f'--db-conn-string "{db_conn_string}" '
        )
        # TODO(WAN): Because the Daemon is a long-running process,
        # it locks .doit.db and prevents other client tasks
        # from executing.
        return f"echo 'Please run: {action}'"

    return {
        "actions": [CmdAction(run_daemon, buffering=1)],
        "verbosity": VERBOSITY_DEFAULT,
        "file_dep": ["./build/bootstrap_pilot.out"],
        "params": [
            # daemon.py parameters.
            {
                "name": "db_conn_string",
                "long": "db_conn_string",
                "help": "Connection string to Pilot database as server.",
                "default": DB_CONN_STRING_PILOT_DAEMON,
            },
        ],
    }


def task_pilot_client():
    """
    Pilot: run the Pilot daemon.
    """

    def client(db_conn_string, command, args):
        action = (
            f"python3 pilot/client.py "
            # client.py arguments.
            f'--db-conn-string "{db_conn_string}" '
            f'--command "{command}" '
            f"{args} "
        )
        return action

    return {
        "actions": [CmdAction(client, buffering=1)],
        "verbosity": VERBOSITY_DEFAULT,
        "uptodate": [False],
        "file_dep": ["./build/bootstrap_pilot.out"],
        "params": [
            # client.py parameters.
            {
                "name": "db_conn_string",
                "long": "db_conn_string",
                "help": "Connection string to Pilot database as client.",
                "default": DB_CONN_STRING_PILOT_CLIENT,
            },
            {
                "name": "command",
                "long": "command",
                "help": "The command to be sent.",
                "default": "NO_COMMAND_SPECIFIED",
            },
            {
                "name": "args",
                "long": "args",
                "help": "Arguments to pass through to client.py.",
                "default": "",
            },
        ],
    }


def task_ci_python():
    """
    CI: this should be run and all warnings fixed before pushing commits.
    """
    folders = ["action", "forecast", "pilot"]

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
