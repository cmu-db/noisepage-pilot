import os
from pathlib import Path

import doit
from doit.action import CmdAction

import dodos.forecast
import dodos.noisepage
from dodos import VERBOSITY_DEFAULT, default_artifacts_path, default_build_path

ARTIFACTS_PATH = default_artifacts_path()
BUILD_PATH = default_build_path()

DEFAULT_DB = "np_as_spiel"
DEFAULT_USER = "np_as_spiel_user"
DEFAULT_PASS = "np_as_spiel_pass"
DB_CONN_STRING_AS_SPIEL = f"host=127.0.0.1 port=5432 dbname={DEFAULT_DB} user={DEFAULT_USER} password={DEFAULT_PASS} sslmode=disable application_name=psql"

# Scratch work.
OPENSPIEL_SRC_PATH = Path("action/selection/open_spiel/").absolute()
OPENSPIEL_BUILD_PATH = (BUILD_PATH / "selection/open_spiel/build/").absolute()
HYPOPG_BUILD_PATH = (BUILD_PATH / "selection/hypopg/build/").absolute()

# Output: predictions.
ARTIFACT_ACTIONS = ARTIFACTS_PATH / "actions.sql"
ARTIFACT_DATABASE_GAME = ARTIFACTS_PATH / "database_game"


def task_action_generation():
    """
    Action generation: generate actions to choose from.
    """
    return {
        "actions": [
            f"mkdir -p {ARTIFACTS_PATH}",
            # Generate create index suggestions for TPC-C.
            f"python3 ./action/generation/generate_create_index_tpcc.py --min-num-cols 1 --max-num-cols 4 --output-sql {ARTIFACT_ACTIONS}",
        ],
        "file_dep": ["./action/generation/generate_create_index_tpcc.py"],
        "targets": [ARTIFACT_ACTIONS],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_action_selection_openspiel_cmake():
    """
    Action selection: invoke CMake for OpenSpiel.

    The build and compile steps are separated for OpenSpiel because this
    repository contains modifications to the OpenSpiel code in the form
    of the database_game.
    """
    return {
        "actions": [
            # Set up directories.
            f"mkdir -p {OPENSPIEL_BUILD_PATH}",
            # Invoke CMake.
            lambda: os.chdir(OPENSPIEL_BUILD_PATH),
            f"cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ {OPENSPIEL_SRC_PATH}",
            # Reset working directory.
            lambda: os.chdir(doit.get_initial_workdir()),
        ],
        "file_dep": [*OPENSPIEL_SRC_PATH.glob("*")],
        "targets": [f"{OPENSPIEL_BUILD_PATH / 'Makefile'}"],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_action_selection_openspiel_build():
    """
    Action selection: build OpenSpiel.
    """
    return {
        "actions": [
            f"mkdir -p {OPENSPIEL_BUILD_PATH}",
            # Build database_game.
            lambda: os.chdir(OPENSPIEL_BUILD_PATH),
            "make -j database_game",
            # Copy the built binary out.
            lambda: os.chdir(doit.get_initial_workdir()),
            f"cp {OPENSPIEL_BUILD_PATH / 'database_game'} {ARTIFACT_DATABASE_GAME}",
            # Reset working directory.
            lambda: os.chdir(doit.get_initial_workdir()),
        ],
        "file_dep": [*OPENSPIEL_SRC_PATH.glob("*"), f"{OPENSPIEL_BUILD_PATH / 'Makefile'}"],
        "targets": [ARTIFACT_DATABASE_GAME],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_action_selection_hypopg_clone():
    """
    Action selection: clone HypoPG for faking index builds.
    """
    return {
        "actions": [
            # Set up directories.
            f"mkdir -p {HYPOPG_BUILD_PATH}",
            lambda: os.chdir(HYPOPG_BUILD_PATH),
            # Clone HypoPG.
            "git clone https://github.com/HypoPG/hypopg.git",
            lambda: os.chdir("hypopg"),
            # Install HypoPG.
            f"sudo PATH={dodos.noisepage.ARTIFACTS_PATH}:$PATH make install",
            # Reset working directory.
            lambda: os.chdir(doit.get_initial_workdir()),
        ],
        "file_dep": [dodos.noisepage.ARTIFACT_pg_config],
        "targets": [HYPOPG_BUILD_PATH / "hypopg/Makefile"],
        "uptodate": [True],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_action_selection_hypopg_install():
    """
    Action selection: install HypoPG for faking index builds.
    """
    return {
        "actions": [
            lambda: os.chdir(HYPOPG_BUILD_PATH / "hypopg"),
            # Install HypoPG.
            f"sudo PATH={dodos.noisepage.ARTIFACTS_PATH}:$PATH make install",
            # Reset working directory.
            lambda: os.chdir(doit.get_initial_workdir()),
        ],
        "file_dep": [HYPOPG_BUILD_PATH / "hypopg/Makefile"],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_action_recommendation_bootstrap_dbms():
    """
    Action recommendation: bootstrap the DBMS with any necessary databases and tables.
    """
    sql_list = [
        f"CREATE USER {DEFAULT_USER} WITH SUPERUSER ENCRYPTED PASSWORD '{DEFAULT_PASS}';",
        f"GRANT ALL PRIVILEGES ON DATABASE {DEFAULT_DB} to {DEFAULT_USER};",
    ]

    return {
        "actions": [
            f"{dodos.noisepage.ARTIFACT_dropdb} --if-exists {DEFAULT_DB}",
            f"{dodos.noisepage.ARTIFACT_dropuser} --if-exists {DEFAULT_USER}",
            f"{dodos.noisepage.ARTIFACT_createdb} {DEFAULT_DB}",
            *[f'{dodos.noisepage.ARTIFACT_psql} --dbname={DEFAULT_DB} -c "{sql}"' for sql in sql_list],
            # Setup HypoPG.
            f'PGPASSWORD={DEFAULT_PASS} {dodos.noisepage.ARTIFACT_psql} --dbname={DEFAULT_DB} -U {DEFAULT_USER} -c "CREATE EXTENSION hypopg;"',
        ],
        "file_dep": [dodos.noisepage.ARTIFACT_psql],
        "verbosity": VERBOSITY_DEFAULT,
        "uptodate": [False],
    }


def task_action_recommendation():
    """
    Action recommendation: apply recommended actions to the DBMS.
    """

    def index_picker(batch_size, db_conn_string):
        action = (
            "python3 ./action/recommendation/index_picker.py "
            # index_picker.py arguments.
            f"--database-game-path {ARTIFACT_DATABASE_GAME} "
            f"--batch-size {batch_size} "
            "-- "
            # database_game arguments.
            f'--db_conn_string "{db_conn_string}" '
            f"--actions_path {ARTIFACT_ACTIONS} "
            f"--forecast_path {dodos.forecast.ARTIFACT_FORECAST} "
        )
        return action

    return {
        "actions": [CmdAction(index_picker)],
        "file_dep": [
            "./action/recommendation/index_picker.py",
            ARTIFACT_ACTIONS,
            ARTIFACT_DATABASE_GAME,
            dodos.forecast.ARTIFACT_FORECAST,
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
