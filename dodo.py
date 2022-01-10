import glob
import os
from pathlib import Path

import doit
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
            "sudo --user postgres psql --file ./bootstrap.sql > build/bootstrap.out 2>&1",
            "sudo --reset-timestamp",
            "echo 'Bootstrap complete; sudo permissions reset.'",
        ],
        "file_dep": ["./bootstrap.sql"],
        "targets": ["./build/bootstrap.out"],
        "verbosity": VERBOSITY_DEFAULT,
    }


##############################################################################
# Third-party dependencies.
##############################################################################


def task_noisepage_build():
    """
    Build NoisePage.
    """

    def build_path():
        return Path("build/noisepage/")

    return {
        "actions": [
            # Set up directories.
            f"mkdir -p {build_path()}",
            # Clone NoisePage.
            f"git clone https://github.com/cmu-db/postgres.git --branch pg14 --single-branch --depth 1 {build_path()}",
            lambda: os.chdir(build_path()),
            # Configure NoisePage.
            "./cmudb/build/configure.sh release",
            # Compile NoisePage.
            "make -j world-bin",
            "make install-world-bin",
            # Move artifacts out.
            lambda: os.chdir(doit.get_initial_workdir()),
            "mkdir -p artifacts/noisepage/",
            f"mv {build_path()}/build/bin/* artifacts/noisepage/",
            "sudo apt-get install --yes bpfcc-tools linux-headers-$(uname -r)",
            f"sudo pip3 install -r {build_path()}/cmudb/tscout/requirements.txt",
            # Reset working directory.
            lambda: os.chdir(doit.get_initial_workdir()),
        ],
        "targets": ["./artifacts/noisepage/postgres"],
        "uptodate": [True],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_benchbase_build():
    """
    Build BenchBase.
    """

    def build_path():
        return Path("build/benchbase/")

    return {
        "actions": [
            # Set up directories.
            f"mkdir -p {build_path()}",
            # Clone BenchBase.
            f"git clone https://github.com/cmu-db/benchbase.git --branch main --single-branch --depth 1 {build_path()}",
            lambda: os.chdir(build_path()),
            # Compile BenchBase.
            "./mvnw clean package -Dmaven.test.skip=true",
            lambda: os.chdir("target"),
            "tar xvzf benchbase-2021-SNAPSHOT.tgz",
            # Move artifacts out.
            lambda: os.chdir(doit.get_initial_workdir()),
            "mkdir -p artifacts/benchbase/",
            f"mv {build_path()}/target/benchbase-2021-SNAPSHOT/* artifacts/benchbase/",
            # Reset working directory.
            lambda: os.chdir(doit.get_initial_workdir()),
        ],
        "targets": ["./artifacts/benchbase/benchbase.jar"],
        "uptodate": [True],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_openspiel_build():
    """
    Action selection: invoke CMake for OpenSpiel.

    The build and compile steps are separated for OpenSpiel because this
    repository contains modifications to the OpenSpiel code in the form of the
    database_game.
    """

    def source_path():
        return Path(doit.get_initial_workdir()) / "action/selection/open_spiel/"

    def build_path():
        return Path("build/action/selection/open_spiel/build/")

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

    def build_path():
        return Path("build/action/selection/open_spiel/build/")

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
        "targets": ["./artifacts/database_game"],
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
            "python3 ./action/generation/generate_create_index_tpcc.py --min-num-cols 1 --max-num-cols 4 --output-sql ./artifacts/actions.sql",
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
            "python3 action/recommendation/index_picker.py "
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


def task_behavior_datagen():
    """
    Behavior modeling: generate training data and perform plan differencing.
    """
    # There should be no scenario in which you do NOT want plan differencing.
    # So it doesn't make sense to expose a separate task just for that.
    training_data_folder = "./artifacts/behavior/datagen/training_data"
    diff_data_folder = "./artifacts/behavior/datagen/differenced_data"

    datagen_args = (
        "--benchbase-user admin "
        "--benchbase-pass password "
        "--config-file ./config/behavior/default.yaml "
        "--dir-benchbase ./artifacts/benchbase/ "
        "--dir-benchbase-config ./config/behavior/benchbase "
        "--dir-noisepage-bin ./artifacts/noisepage "
        "--dir-tscout ./build/noisepage/cmudb/tscout/ "
        f"--dir-output {training_data_folder} "
        "--dir-tmp ./build/behavior/datagen/ "
        "--path-noisepage-conf ./config/behavior/postgres/postgresql.conf "
        "--tscout-wait-sec 2 "
    )
    datadiff_args = (
        f"--dir-datagen-data {training_data_folder} --dir-output {diff_data_folder} "
    )

    return {
        "actions": [
            # "sudo --validate",
            f"mkdir -p {training_data_folder}",
            f"mkdir -p {diff_data_folder}",
            "mkdir -p build/behavior/datagen/",
            f"python3 -m behavior datagen {datagen_args}",
            "sudo --reset-timestamp",
            # Immediately perform plan differencing here since the models
            # suck without differencing anyway.
            f"python3 -m behavior datadiff {datadiff_args}",
        ],
        "file_dep": [
            "./artifacts/benchbase/benchbase.jar",
            "./artifacts/noisepage/postgres",
        ],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_behavior_train():
    """
    Behavior modeling: train OU models.
    """
    output_folder = "./artifacts/behavior/models"
    train_args = (
        "--config-file ./config/behavior/default.yaml "
        "--dir-data-train ./artifacts/behavior/datagen/differenced_data/diff/train/ "
        "--dir-data-eval ./artifacts/behavior/datagen/differenced_data/diff/eval/ "
        f"--dir-output {output_folder} "
    )

    return {
        "actions": [
            f"mkdir -p {output_folder}",
            f"python3 -m behavior train {train_args}",
        ],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_ci_python():
    """
    CI: this should be run and all warnings fixed before pushing commits.
    """
    folders = ["action", "behavior", "forecast", "pilot"]

    return {
        "actions": [
            "black --verbose dodo.py setup.py",
            "isort --verbose dodo.py setup.py",
            "flake8 --statistics dodo.py setup.py",
            *[f"black --verbose {folder}" for folder in folders],
            *[f"isort {folder}" for folder in folders],
            *[f"flake8 --statistics {folder}" for folder in folders],
            # TODO(WAN): Only run pylint on behavior for now.
            *[f"pylint --verbose {folder}" for folder in ["behavior"]],
        ],
        "verbosity": VERBOSITY_DEFAULT,
        "uptodate": [False],
    }
