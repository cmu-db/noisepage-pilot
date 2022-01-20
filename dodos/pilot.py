from doit.action import CmdAction
from plumbum import local

import dodos.noisepage
from dodos import VERBOSITY_DEFAULT

DEFAULT_DB = "np_pilot"
DEFAULT_USER_DAEMON = "np_pilot_daemon"
DEFAULT_USER_CLIENT = "np_pilot_client"
DEFAULT_PASS_DAEMON = "np_pilot_daemon_pass"
DEFAULT_PASS_CLIENT = "np_pilot_client_pass"

DB_CONN_STRING_PILOT_DAEMON = f"host=127.0.0.1 port=5432 dbname={DEFAULT_DB} user={DEFAULT_USER_DAEMON} password={DEFAULT_PASS_DAEMON} sslmode=disable application_name=psql"
DB_CONN_STRING_PILOT_CLIENT = f"host=127.0.0.1 port=5432 dbname={DEFAULT_DB} user={DEFAULT_USER_CLIENT} password={DEFAULT_PASS_CLIENT} sslmode=disable application_name=psql"


def task_pilot_bootstrap():
    """
    Pilot: bootstrap the Pilot tables.
    """

    sql_list = [
        f"CREATE USER {DEFAULT_USER_DAEMON} WITH ENCRYPTED PASSWORD '{DEFAULT_PASS_DAEMON}';",
        f"GRANT ALL PRIVILEGES ON DATABASE {DEFAULT_DB} to {DEFAULT_USER_DAEMON};",
        f"CREATE USER {DEFAULT_USER_CLIENT} WITH ENCRYPTED PASSWORD '{DEFAULT_PASS_CLIENT}';",
        f"GRANT ALL PRIVILEGES ON DATABASE {DEFAULT_DB} to {DEFAULT_USER_CLIENT};",
    ]

    def bootstrap(db_conn_string):
        action = f'psql "{db_conn_string}" --file ./pilot/bootstrap_tables.sql'
        return action

    return {
        "actions": [
            f"{dodos.noisepage.ARTIFACT_dropdb} --if-exists {DEFAULT_DB}",
            f"{dodos.noisepage.ARTIFACT_dropuser} --if-exists {DEFAULT_USER_DAEMON}",
            f"{dodos.noisepage.ARTIFACT_dropuser} --if-exists {DEFAULT_USER_CLIENT}",
            f"{dodos.noisepage.ARTIFACT_createdb} {DEFAULT_DB}",
            *[f'{dodos.noisepage.ARTIFACT_psql} --dbname={DEFAULT_DB} -c "{sql}"' for sql in sql_list],
            # Bootstrap additional pilot tables.
            CmdAction(bootstrap),
        ],
        "verbosity": VERBOSITY_DEFAULT,
        "file_dep": ["./pilot/bootstrap_tables.sql"],
        "uptodate": [False],
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
    Pilot: run the Pilot daemon detached.
    """

    def run_daemon_detached(db_conn_string):
        action = local["python3"][
            "pilot/daemon.py",
            "--db-conn-string",
            f"{db_conn_string}",
        ]
        ret = action.run_nohup(stdout="pilot_daemon.out")
        print(f"Pilot Daemon: {ret.pid}")

    return {
        "actions": [run_daemon_detached],
        "verbosity": VERBOSITY_DEFAULT,
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
    Pilot: send a command to the Pilot.
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
