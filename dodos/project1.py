from plumbum import cmd
from dodos import VERBOSITY_DEFAULT

DEFAULT_DB="project1db"
DEFAULT_USER="project1user"
DEFAULT_PASS="project1pass"

def task_project1_enable_logging():
    """
    Project1: enable logging. (will cause a restart)
    """
    sql_list = [
        "ALTER SYSTEM SET log_destination='csvlog'",
        "ALTER SYSTEM SET logging_collector='on'",
        "ALTER SYSTEM SET log_statement='all'",
    ]

    return {
        "actions": [
            *[
                f'PGPASSWORD={DEFAULT_PASS} psql --host=localhost --dbname={DEFAULT_DB} --username={DEFAULT_USER} --command="{sql}"'
                for sql in sql_list
            ],
            lambda: cmd.sudo["systemctl"]["restart", "postgresql"].run_fg(),
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
    ]

    return {
        "actions": [
            *[
                f'PGPASSWORD={DEFAULT_PASS} psql --host=localhost --dbname={DEFAULT_DB} --username={DEFAULT_USER} --command="{sql}"'
                for sql in sql_list
            ],
            lambda: cmd.sudo["systemctl"]["restart", "postgresql"].run_fg(),
        ],
        "verbosity": VERBOSITY_DEFAULT,
    }
