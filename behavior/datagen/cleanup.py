#!/usr/bin/env python3

# Note: This script needs root permissions to work
import argparse
import shutil
from pathlib import Path

import psutil


def kill_tscout_and_postgres() -> None:
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


def chown_results(username: str) -> None:
    # change the tscout results ownership to the user who ran the benchmark
    results_dir = Path(f"/home/{username}/noisepage-pilot/data/behavior/training_data/")

    if results_dir.exists():
        print(f"Changing ownership of TScout results from root to user: {username}")
        shutil.chown(str(results_dir), user=username)
        for file in results_dir.glob("**/*"):
            shutil.chown(file, user=username)
        print("Cleanup Complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Postgres/Benchbase/TScout")
    parser.add_argument(
        "--username", help="Username to reassign file ownership", required=False
    )
    args = parser.parse_args()

    kill_tscout_and_postgres()

    if args.username is not None:
        chown_results(args.username)
    else:
        print("No username provided, cannot reassign result file ownership.")
