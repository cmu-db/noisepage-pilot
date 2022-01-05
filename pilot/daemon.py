# This file defines the Pilot daemon.
# The Pilot daemon handles logic for controlling the explorer, etc.
#
# Before modifying this file, please read:
# https://www.postgresql.org/docs/14/sql-notify.html
# https://www.postgresql.org/docs/14/sql-listen.html
# https://www.psycopg.org/psycopg3/docs/advanced/async.html#async-notify


import select

import protocol
import psycopg
from plumbum import cli


def _handle(event):
    """
    Handle the NOTIFY event.

    Parameters
    ----------
    event : protocol.NotifyEvent
        The event to be handled.
    """
    print(event)
    pass


class DaemonCLI(cli.Application):
    db_conn_string = cli.SwitchAttr("--db-conn-string", str, mandatory=True)
    channel_name = cli.SwitchAttr("--channel-name", str, default="pilot")
    timeout_sec = cli.SwitchAttr("--timeout-sec", int, default=60)

    def main(self):
        # autocommit is required because of how NOTIFY interacts with SQL transactions.
        with psycopg.connect(self.db_conn_string, autocommit=True) as conn:
            cursor = conn.cursor(row_factory=psycopg.rows.dict_row)
            cursor.execute(f"LISTEN {self.channel_name};")
            print(f"Listening (timeout {self.timeout_sec} s): {self.channel_name}")

            while True:
                # Wait for a NOTIFY message to be received.
                if select.select([conn], [], [], self.timeout_sec) == ([], [], []):
                    # A NOTIFY message was not received and we timed out.
                    # TODO(WAN): Is there any use case for the Explorer timing out?
                    #  e.g., non-critical maintenance tasks to be run.
                    pass
                else:
                    # NOTIFY was received. Process all the NOTIFY messages.
                    # psycopg's add_notify_handler is NOT used because the handler
                    # only fires during a connection operation.
                    for notify in conn.notifies():
                        event = protocol.Server.notify_recv(notify)
                        _handle(event)


if __name__ == "__main__":
    DaemonCLI.run()
