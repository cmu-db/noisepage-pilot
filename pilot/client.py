import csv
from io import StringIO

from plumbum import cli
from protocol import Client


class ClientCLI(cli.Application):
    db_conn_string = cli.SwitchAttr(
        "--db-conn-string",
        str,
        mandatory=True,
        help="Connection string to Pilot database.",
    )
    command = cli.SwitchAttr(
        "--command",
        str,
        mandatory=True,
        help="The Pilot Daemon command to be invoked.",
    )

    def main(self, *args):
        # Get the function to be invoked that will
        # send a command to the Pilot Daemon.
        client = Client(self.db_conn_string)
        func = client.get_function(self.command)

        # Get the arguments to the function, if any.
        data = {}
        assert len(args) in [0, 1]
        if len(args) == 1:
            # TODO(WAN): We assume a CSV string of input.
            iostr = StringIO(args[0])
            reader = csv.reader(iostr)
            data = {key: val for row in reader for item in row for key, val in [item.split("=", maxsplit=1)]}

        # Invoke the function.
        func(data)


if __name__ == "__main__":
    ClientCLI.run()
