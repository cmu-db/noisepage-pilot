import os
from pathlib import Path

import doit
from doit.action import CmdAction
from plumbum import local

from dodos import VERBOSITY_DEFAULT, default_artifacts_path, default_build_path

ARTIFACTS_PATH = default_artifacts_path()
BUILD_PATH = default_build_path()
DEFAULT_POSTGRESQL_CONF_PATH = Path("config/postgres/default_postgresql.conf").absolute()

DEFAULT_DB = "noisepage"
DEFAULT_USER = "terrier"
DEFAULT_PASS = "woof"
DEFAULT_PGDATA = "pgdata"

# Output: various useful binaries and folders.
ARTIFACT_createdb = ARTIFACTS_PATH / "createdb"
ARTIFACT_createuser = ARTIFACTS_PATH / "createuser"
ARTIFACT_dropdb = ARTIFACTS_PATH / "dropdb"
ARTIFACT_dropuser = ARTIFACTS_PATH / "dropuser"
ARTIFACT_pg_config = ARTIFACTS_PATH / "pg_config"
ARTIFACT_pg_ctl = ARTIFACTS_PATH / "pg_ctl"
ARTIFACT_postgres = ARTIFACTS_PATH / "postgres"
ARTIFACT_psql = ARTIFACTS_PATH / "psql"
ARTIFACT_pgdata = ARTIFACTS_PATH / DEFAULT_PGDATA
ARTIFACT_pgdata_log = ARTIFACTS_PATH / DEFAULT_PGDATA / "log"


def task_noisepage_clone():
    """
    NoisePage: clone.
    """

    def repo_clone(repo_url):
        cmd = f"git clone {repo_url} --branch pg14 --single-branch --depth 1 {BUILD_PATH}"
        return cmd

    return {
        "actions": [
            # Set up directories.
            f"mkdir -p {BUILD_PATH}",
            # Clone NoisePage.
            CmdAction(repo_clone),
            # Reset working directory.
            lambda: os.chdir(doit.get_initial_workdir()),
        ],
        "targets": [BUILD_PATH, BUILD_PATH / "Makefile"],
        "uptodate": [True],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "repo_url",
                "long": "repo_url",
                "help": "The repository to clone from.",
                "default": "https://github.com/cmu-db/postgres.git",
            },
        ],
    }


def task_noisepage_build():
    """
    NoisePage: build.
    """
    return {
        "actions": [
            lambda: os.chdir(BUILD_PATH),
            # Configure NoisePage.
            "doit np_config --build_type=release",
            # Compile NoisePage.
            "doit np_build",
            # Move artifacts out.
            lambda: os.chdir(doit.get_initial_workdir()),
            f"mkdir -p {ARTIFACTS_PATH}",
            f"cp {BUILD_PATH / 'build/bin/*'} {ARTIFACTS_PATH}",
            "sudo apt-get install --yes bpfcc-tools linux-headers-$(uname -r)",
            f"sudo pip3 install -r {BUILD_PATH / 'cmudb/tscout/requirements.txt'}",
            # Reset working directory.
            lambda: os.chdir(doit.get_initial_workdir()),
        ],
        "file_dep": [BUILD_PATH / "Makefile"],
        "targets": [
            ARTIFACTS_PATH,
            ARTIFACT_createdb,
            ARTIFACT_createuser,
            ARTIFACT_dropdb,
            ARTIFACT_dropuser,
            ARTIFACT_pg_config,
            ARTIFACT_pg_ctl,
            ARTIFACT_postgres,
            ARTIFACT_psql,
        ],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_noisepage_init():
    """
    NoisePage: run NoisePage in detached mode.
    """

    def run_noisepage_detached(config):
        local["cp"][f"{config}", f"{DEFAULT_PGDATA}/postgresql.conf"].run_nohup()
        ret = local["./pg_ctl"]["start", "-D", DEFAULT_PGDATA].run_nohup(stdout="noisepage.out")
        print(f"NoisePage PID: {ret.pid}")

    sql_list = [
        f"CREATE ROLE {DEFAULT_USER} WITH LOGIN SUPERUSER ENCRYPTED PASSWORD '{DEFAULT_PASS}'",
    ]

    return {
        "actions": [
            "pkill postgres || true",
            lambda: os.chdir(ARTIFACTS_PATH),
            f"rm -rf {DEFAULT_PGDATA}",
            f"./initdb {DEFAULT_PGDATA}",
            run_noisepage_detached,
            "until ./pg_isready ; do sleep 1 ; done",
            f"./createdb {DEFAULT_DB}",
            *[f'./psql --dbname={DEFAULT_DB} --command="{sql}"' for sql in sql_list],
            # Reset working directory.
            lambda: os.chdir(doit.get_initial_workdir()),
        ],
        "file_dep": [ARTIFACT_pg_ctl],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "config",
                "long": "config",
                "help": "Path to the postgresql.conf configuration file.",
                "default": DEFAULT_POSTGRESQL_CONF_PATH,
            },
        ],
    }


def task_noisepage_swap_config():
    """
    NoisePage: swaps the postgresql.conf for the current running instance.
    """

    def swap_config(config):
        local["./pg_ctl"]["stop", "-D", DEFAULT_PGDATA, "-m", "smart"].run(retcode=None)
        local["cp"][f"{config}", f"{DEFAULT_PGDATA}/postgresql.conf"].run_nohup()
        ret = local["./pg_ctl"]["start", "-D", DEFAULT_PGDATA].run_nohup(stdout="noisepage.out")
        print(f"NoisePage PID: {ret.pid}")

    return {
        "actions": [
            lambda: os.chdir(ARTIFACTS_PATH),
            swap_config,
            "until ./pg_isready ; do sleep 1 ; done",
            lambda: os.chdir(doit.get_initial_workdir()),
        ],
        "file_dep": [ARTIFACT_pg_ctl],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "config",
                "long": "config",
                "help": "Path to the postgresql.conf configuration file.",
                "default": DEFAULT_POSTGRESQL_CONF_PATH,
            },
        ],
    }


def task_noisepage_hutch_install():
    """
    NoisePage: install hutch extension to support EXPLAIN (format tscout).
    """
    sql_list = [
        # Note that this will overwrite any existing settings of shared_preload_libraries.
        "ALTER SYSTEM SET shared_preload_libraries='hutch_extension'",
    ]

    return {
        "actions": [
            lambda: os.chdir(BUILD_PATH),
            # Compile and install Hutch.
            "doit hutch_install",
            lambda: os.chdir(ARTIFACTS_PATH),
            *[
                f'PGPASSWORD={DEFAULT_PASS} ./psql --dbname={DEFAULT_DB} --username={DEFAULT_USER} --command="{sql}"'
                for sql in sql_list
            ],
            lambda: local["./pg_ctl"]["restart", "-D", DEFAULT_PGDATA].run_fg(),
            # Reset working directory.
            lambda: os.chdir(doit.get_initial_workdir()),
        ],
        "uptodate": [False],
        "file_dep": [ARTIFACT_postgres],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_noisepage_enable_logging():
    """
    NoisePage: enable logging. (will cause a restart)
    """
    sql_list = [
        "ALTER SYSTEM SET log_destination='csvlog'",
        "ALTER SYSTEM SET log_statement='all'",
        "ALTER SYSTEM SET logging_collector=on",
    ]

    return {
        "actions": [
            lambda: os.chdir(ARTIFACTS_PATH),
            *[
                f'PGPASSWORD={DEFAULT_PASS} ./psql --dbname={DEFAULT_DB} --username={DEFAULT_USER} --command="{sql}"'
                for sql in sql_list
            ],
            lambda: local["./pg_ctl"]["restart", "-D", DEFAULT_PGDATA].run_fg(),
            # Reset working directory.
            lambda: os.chdir(doit.get_initial_workdir()),
        ],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_noisepage_disable_logging():
    """
    NoisePage: disable logging. (will cause a restart)
    """
    sql_list = [
        "ALTER SYSTEM SET log_destination='stderr'",
        "ALTER SYSTEM SET log_statement='none'",
        "ALTER SYSTEM SET logging_collector=off",
    ]

    return {
        "actions": [
            lambda: os.chdir(ARTIFACTS_PATH),
            *[
                f'PGPASSWORD={DEFAULT_PASS} ./psql --dbname={DEFAULT_DB} --username={DEFAULT_USER} --command="{sql}"'
                for sql in sql_list
            ],
            lambda: local["./pg_ctl"]["restart", "-D", DEFAULT_PGDATA].run_fg(),
            # Reset working directory.
            lambda: os.chdir(doit.get_initial_workdir()),
        ],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_noisepage_truncate_log():
    """
    NoisePage: truncate the most recent query log files.
    """

    return {
        "actions": [
            lambda: os.chdir(ARTIFACT_pgdata_log),
            'RECENT=$(ls --sort=time --reverse *.csv | head --lines=%(num_files)s) ; echo "Deleting:\n$RECENT" ; rm $RECENT ; echo "Deleted."',
            # Reset working directory.
            lambda: os.chdir(doit.get_initial_workdir()),
        ],
        "verbosity": VERBOSITY_DEFAULT,
        "uptodate": [False],
        "params": [
            {
                "name": "num_files",
                "long": "num_files",
                "help": "The number of query log files to remove.",
                "default": 5,
            },
        ],
    }
