import os
import sys
import time
from pathlib import Path

from plumbum import cmd, local

from dodos import VERBOSITY_DEFAULT
from dodos.noisepage import BUILD_PATH, ARTIFACT_pg_ctl


def task_tscout_init():
    """
    TScout: attach TScout to a running NoisePage instance
    """

    def start_tscout(output_dir, wait_time):
        if output_dir is None:
            print("Unable to start tscout without an output directory")
            return False

        postmaster_pid = local["pgrep"]["-ox", "postgres"]()
        dir_tscout_output = Path(output_dir) / "tscout"
        dir_tscout_output.mkdir(parents=True, exist_ok=True)
        dir_tscout_output = dir_tscout_output.absolute()
        print("Attaching TScout.")

        dir_tscout = BUILD_PATH / "cmudb/tscout"
        os.chdir(dir_tscout)
        cmd.sudo["python3"]["tscout.py", postmaster_pid, "--outdir", dir_tscout_output].run_bg(
            # sys.stdout will actually give the doit writer. Here we need the actual
            # underlying output stream.
            stdout=sys.__stdout__,
            stderr=sys.__stderr__,
        )

        time.sleep(wait_time)

    return {
        "actions": [start_tscout],
        "file_dep": [ARTIFACT_pg_ctl],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "output_dir",
                "long": "output_dir",
                "help": "Directory that tscout should output to",
                "default": None,
            },
            {
                "name": "wait_time",
                "long": "wait_time",
                "help": "Time to wait (seconds) after TScout has been started.",
                "default": 5,
            },
        ],
    }


def task_tscout_shutdown():
    """
    TScout: shutdown the running TScout instance
    """

    def shutdown_tscout(output_dir, wait_time):
        if output_dir is None:
            print("Unable to start tscout without an output directory")
            return False

        cmd.sudo["pkill", "-i", "tscout"](retcode=None)
        time.sleep(wait_time)

        # Because TScout has to execute with sudo,
        # the results are owned by root.
        # Take ownership of TScout's results.
        owner = Path(__file__).owner()
        training_data_dir = Path(output_dir)
        if training_data_dir.exists():
            print(f"Taking ownership of: {training_data_dir}")
            cmd.sudo["chown", "--recursive", owner, training_data_dir]()

        # Nuke that irritating psycache folder.
        dir_tscout = BUILD_PATH / "cmudb/tscout/__pycache__"
        cmd.sudo["rm", "-rf", dir_tscout]()

    return {
        "actions": [shutdown_tscout],
        "file_dep": [ARTIFACT_pg_ctl],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "output_dir",
                "long": "output_dir",
                "help": "Directory that tscout should output to",
                "default": None,
            },
            {
                "name": "wait_time",
                "long": "wait_time",
                "help": "Time to wait (seconds) before shutting down TScout.",
                "default": 5,
            },
        ],
    }
