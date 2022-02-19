from plumbum import cmd

from dodos import VERBOSITY_DEFAULT

DEFAULT_DB = "project1db"
DEFAULT_USER = "project1user"
DEFAULT_PASS = "project1pass"

# Note that pgreplay requires the following configuration:
#
# log_min_messages=error (or more)
# log_min_error_statement=log (or more)
# log_connections=on
# log_disconnections=on
# log_line_prefix='%m|%u|%d|%c|' (if you don't use CSV logging)
# log_statement='all'
# lc_messages must be set to English (encoding does not matter)
# bytea_output=escape (from version 9.0 on, only if you want to replay the log on 8.4 or earlier)
#
# Additionally, doit has a bit of an anti-feature with command substitution,
# so you have to escape %'s by Python %-formatting rules (no way to disable this behavior).


def task_project1_enable_logging():
    """
    Project1: enable logging. (will cause a restart)
    """
    sql_list = [
        "ALTER SYSTEM SET log_destination='csvlog'",
        "ALTER SYSTEM SET logging_collector='on'",
        "ALTER SYSTEM SET log_statement='all'",
        # For pgreplay.
        "ALTER SYSTEM SET log_connections='on'",
        "ALTER SYSTEM SET log_disconnections='on'",
    ]

    return {
        "actions": [
            *[
                f'PGPASSWORD={DEFAULT_PASS} psql --host=localhost --dbname={DEFAULT_DB} --username={DEFAULT_USER} --command="{sql}"'
                for sql in sql_list
            ],
            lambda: cmd.sudo["systemctl"]["restart", "postgresql"].run_fg(),
            "until pg_isready ; do sleep 1 ; done",
        ],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_project1_disable_logging():
    """
    Project1: disable logging. (will cause a restart)
    """
    sql_list = [
        "ALTER SYSTEM SET log_destination='stderr'",
        "ALTER SYSTEM SET logging_collector='off'",
        "ALTER SYSTEM SET log_statement='none'",
        # For pgreplay.
        "ALTER SYSTEM SET log_connections='off'",
        "ALTER SYSTEM SET log_disconnections='off'",
    ]

    return {
        "actions": [
            *[
                f'PGPASSWORD={DEFAULT_PASS} psql --host=localhost --dbname={DEFAULT_DB} --username={DEFAULT_USER} --command="{sql}"'
                for sql in sql_list
            ],
            lambda: cmd.sudo["systemctl"]["restart", "postgresql"].run_fg(),
            "until pg_isready ; do sleep 1 ; done",
        ],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_project1_reset_db():
    """
    Project1: drop (if exists) and create project1db.
    """

    return {
        "actions": [
            # Drop the project database if it exists.
            f"PGPASSWORD={DEFAULT_PASS} dropdb --host=localhost --username={DEFAULT_USER} --if-exists {DEFAULT_DB}",
            # Create the project database.
            f"PGPASSWORD={DEFAULT_PASS} createdb --host=localhost --username={DEFAULT_USER} {DEFAULT_DB}",
            "until pg_isready ; do sleep 1 ; done",
        ],
        "verbosity": VERBOSITY_DEFAULT,
    }
