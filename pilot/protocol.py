# This file defines the protocol for messages sent over LISTEN/NOTIFY.
# The constants in this file must be kept in sync with bootstrap_tables.sql.

# Before modifying this file, please read:
# - https://www.psycopg.org/psycopg3/docs/basic/params.html

import enum
import json
from dataclasses import dataclass
from typing import Dict

import psycopg


class CmdType(enum.Enum):
    """
    The possible types of commands.
    This must be kept in sync with pilot_cmd_type.
    """

    RUN = "cmd_run"
    ABORT = "cmd_abort"


@dataclass
class NotifyEvent:
    """
    A parsed NOTIFY event.
    """

    pid: int  # The server process PID of the NOTIFY session.
    channel: str  # The channel which the notify came on.
    cmd_id: str  # The command ID.
    cmd_type: CmdType  # The command type.
    args: Dict  # Any additional data sent in the notify.


class Server:
    """
    Wrapper for server methods for the Pilot component.
    """

    @staticmethod
    def notify_recv(notify):
        """
        Parse the raw NOTIFY event received by psycopg.

        Parameters
        ----------
        notify : psycopg.Notify
            The notification received by psycopg.

        Returns
        -------
        event : NotifyEvent
            The notify event that was received.
        """
        cmd_id, cmd_type, args_str = notify.payload.split(",", maxsplit=2)
        event = NotifyEvent(
            notify.pid,
            notify.channel,
            cmd_id,
            CmdType(cmd_type),
            json.loads(args_str),
        )
        return event


class Client:
    """
    Wrapper for Client methods for the Pilot component.
    """

    def __init__(self, db_conn_string):
        """
        Create a new Client instance.

        Parameters
        ----------
        db_conn_string : str
            Connection string to the Pilot database.
        """
        self.db_conn_string = db_conn_string

    def _notify_send(self, ctype, **kwargs):
        """
        Send a notify from the client to the Pilot.

        Parameters
        ----------
        ctype : CmdType
            Any command recognized by the Pilot.
        kwargs : str
            Any number of keyword arguments to the specified command.
        """
        with psycopg.connect(self.db_conn_string) as conn:
            with conn.cursor() as cursor:
                sql = (
                    "INSERT INTO "
                    "pilot_commands (ctype, args)"
                    "VALUES "
                    "(%(ctype)s, %(args)s)"
                )
                params = {"ctype": ctype.value, "args": json.dumps(kwargs)}
                cursor.execute(sql, params)

    def get_function(self, cli_arg):
        """
        Get the specified convenience function,
        or construct one on the fly.

        Parameters
        ----------
        cli_arg : str
            The name of a convenience function,
            or the name of the NOTIFY event to invoke.

        Returns
        -------
        func : Callable[Dict, None]
            The convenience function to be invoked with a dict of arguments.
        """
        # TODO(WAN): Complicated NOTIFY chains can go here.
        def new_func(args):
            self._notify_send(CmdType.RUN, command=cli_arg, **args)

        return new_func
