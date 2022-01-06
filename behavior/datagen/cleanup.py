#!/usr/bin/env python3

# Note: This script needs root permissions to work
import shutil
from pathlib import Path

import psutil


def main() -> None:
    # First kill TScout and Postgres then claim ownership of any result files
    print("Shutting down TScout and Postgres")
    tscout_process_names = [
        "TScout Coordinator",
        "TScout Processor",
        "TScout Collector",
    ]

    try:
        for proc in psutil.process_iter(["name", "pid"]):

            if "postgres" in proc.info["name"].lower():
                try:
                    proc.kill()
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    pass
            elif any(
                (
                    tscout_process_name in proc.info["name"]
                    for tscout_process_name in tscout_process_names
                )
            ):
                try:
                    proc.kill()
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    pass
        print("Shutdown TScout and Postgres successfully")
    except RuntimeError as err:
        print(f"Error shutting down TScout and Postgres: {err}")

    # change the tscout results ownership from root to the correct user
    owner = Path(__file__).owner()
    training_data_dir = Path(
        f"/home/{owner}/noisepage-pilot/data/behavior/training_data"
    )
    if training_data_dir.exists():
        print(f"Changing ownership of TScout results from root to user: {owner}")
        shutil.chown(training_data_dir, user=owner)
        for file in training_data_dir.glob("**/*"):
            shutil.chown(file, user=owner)
        print("Cleanup Complete")


if __name__ == "__main__":
    main()
