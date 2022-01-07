# allow plumbum FG statements
# pylint: disable=pointless-statement
from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import psutil
import yaml
from plumbum import BG, FG, ProcessExecutionError, local
from plumbum.cmd import sudo  # pylint: disable=import-error

from behavior import (
    BEHAVIOR_DATA_DIR,
    BENCHBASE_CONFIG_DIR,
    BENCHBASE_DIR,
    BENCHDB_TO_TABLES,
    CONFIG_DIR,
    PG_CONFIG_DIR,
    PG_DIR,
    SQLSMITH_DIR,
    TSCOUT_DIR,
)
from behavior.util import init_logging


def check_orphans() -> None:
    """Check for TScout and Postgres processes from prior runs, as they cause the runner to fail.

    This will throw an error if it finds *any* postgres processes.
    """

    tscout_process_names = [
        "TScout Coordinator",
        "TScout Processor",
        "TScout Collector",
    ]
    pg_procs = []
    tscout_procs = []

    for proc in psutil.process_iter(["pid", "name", "username", "ppid", "create_time"]):
        proc_name = proc.info["name"].lower()
        if "postgres" in proc_name:
            pg_procs.append(proc)

        if any(
            (
                tscout_process_name in proc.info["name"]
                for tscout_process_name in tscout_process_names
            )
        ):
            tscout_procs.append(proc)

    assert (
        len(pg_procs) == 0
    ), f"Found active postgres processes from previous runs: {pg_procs}"
    assert (
        len(tscout_procs) == 0
    ), f"Found active tscout processes from previous runs: {tscout_procs}"


def query(db: str, _query: str, args: Optional[list[str]] = None) -> None:
    psql_path = str(PG_DIR / "build" / "bin" / "psql")
    psql = local[psql_path]

    if args is not None:
        psql["-d", db, "-c", _query, args]()
    else:
        psql["-d", db, "-c", _query]()


def init_pg(config: dict[str, Any]) -> None:
    """Launch Postgres for a single benchmark run."""

    try:
        os.chdir(PG_DIR)

        # initialize postgres for benchbase execution
        pg_data_dir = PG_DIR / "data"
        if pg_data_dir.exists():
            logging.debug("Removing existing Postgres data directory: %s", pg_data_dir)
            shutil.rmtree(pg_data_dir)

        pg_data_dir.mkdir(parents=True, exist_ok=True)
        pg_ctl = local["./build/bin/pg_ctl"]
        pg_ctl["initdb", "-D", "data"]()
        shutil.copy(PG_CONFIG_DIR / "postgresql.conf", pg_data_dir / "postgresql.conf")
        pg_ctl["-D", "data", "-o", "-W 2", "start"] & FG

        # Initialize the DB and create an admin user for Benchbase to use
        createdb = local["./build/bin/createdb"]
        createdb["test"]()
        query(
            "test",
            "CREATE ROLE admin WITH PASSWORD 'password' SUPERUSER CREATEDB CREATEROLE INHERIT LOGIN;",
        )
        createdb["-O", "admin", "benchbase"]()
        query("test", "ALTER DATABASE test SET compute_query_id = 'ON';")
        query("benchbase", "ALTER DATABASE benchbase SET compute_query_id = 'ON';")

        if config["auto_explain"]:
            query("benchbase", "ALTER SYSTEM SET auto_explain.log_min_duration = 0;")
            pg_ctl["-D", "data", "reload"] & FG

        if config["pg_stat_statements"]:
            query("benchbase", "CREATE EXTENSION pg_stat_statements;")

        if config["pg_store_plans"]:
            query("benchbase", "CREATE EXTENSION pg_store_plans;")

        # Turn off pager to avoid needing tty interaction
        query("benchbase", "SELECT 1;", ["-P", "pager=off"])

    except (KeyboardInterrupt, FileNotFoundError, ProcessExecutionError) as err:
        cleanup(err, terminate=True, message="Error initializing Postgres")


def pg_analyze(bench_db: str) -> None:
    """Analyze all tables in the specific benchmark database.  Updates Postgres' internal statistics for more accurate cardinality estimation and plan costing."""
    try:
        logging.info("Analyzing tables")
        for table in BENCHDB_TO_TABLES[bench_db]:
            query("benchbase", f"ANALYZE VERBOSE {table};")
    except ProcessExecutionError as err:
        cleanup(err, terminate=True, message="Error analyzing Postgres")


def pg_prewarm(bench_db: str) -> None:
    """Prewarm Postgres so the buffer pool and OS page cache has the workload data available."""

    try:
        logging.info("Prewarming tables")
        query("benchbase", "CREATE EXTENSION pg_prewarm;")
        for table in BENCHDB_TO_TABLES[bench_db]:
            query("benchbase", f"SELECT * from pg_prewarm('{table}');")
    except (KeyboardInterrupt, FileNotFoundError, ProcessExecutionError) as err:
        cleanup(err, terminate=True, message="Error prewarming Postgres")


def init_tscout(results_dir: Path) -> None:
    """Launch TScout"""

    try:
        os.chdir(TSCOUT_DIR)
        tscout_results_dir = results_dir / "tscout"
        tscout_results_dir.mkdir(exist_ok=True)

        # assumes the oldest Postgres PID is the Postmaster
        postmaster_pid = local["pgrep"]["-ox", "postgres"]()
        (
            sudo["python3"]["tscout.py", postmaster_pid, "--outdir", tscout_results_dir]
            & BG
        )
    except (FileNotFoundError, ProcessExecutionError) as err:
        cleanup(err, terminate=True, message="Error initializing TScout")

    time.sleep(5)  # allows tscout to attach before Benchbase execution begins


def init_benchbase(bench_db: str, benchbase_results_dir: Path) -> None:
    """Initialize Benchbase and load benchmark data."""

    try:
        benchbase_snapshot_dir = BENCHBASE_DIR / "benchbase-2021-SNAPSHOT"
        os.chdir(benchbase_snapshot_dir)

        # move runner config to benchbase and also save it in the output directory
        input_cfg_path = BENCHBASE_CONFIG_DIR / f"{bench_db}_config.xml"
        benchbase_cfg_path = (
            benchbase_snapshot_dir / f"config/postgres/{bench_db}_config.xml"
        )
        shutil.copy(input_cfg_path, benchbase_cfg_path)
        shutil.copy(input_cfg_path, benchbase_results_dir)

        logging.info("Initializing Benchbase for DB: %s", bench_db)
        benchbase_cmd = [
            "-jar",
            "benchbase.jar",
            "-b",
            bench_db,
            "-c",
            f"config/postgres/{bench_db}_config.xml",
            "--create=true",
            "--load=true",
            "--execute=false",
        ]
        local["java"][benchbase_cmd]()
        logging.info("Initialized Benchbase for Benchmark: %s", bench_db)
    except (KeyboardInterrupt, FileNotFoundError, ProcessExecutionError) as err:
        cleanup(err, terminate=True, message="Error initializing Benchbase")


def exec_benchbase(
    bench_db: str,
    results_dir: Path,
    benchbase_results_dir: Path,
    config: dict[str, Any],
) -> None:
    """Run Benchbase on the specific benchmark database."""
    try:
        if config["pg_stat_statements"]:
            query("benchbase", "SELECT pg_stat_statements_reset();")

        if config["pg_store_plans"]:
            query("benchbase", "SELECT pg_store_plans_reset();")

        benchbase_snapshot_dir = BENCHBASE_DIR / "benchbase-2021-SNAPSHOT"
        os.chdir(benchbase_snapshot_dir)
        logging.info("Begin Benchbase execution for benchmark: %s", bench_db)
        args = [
            "-jar",
            "benchbase.jar",
            "-b",
            bench_db,
            "-c",
            f"config/postgres/{bench_db}_config.xml",
            "--create=false",
            "--load=false",
            "--execute=true",
        ]
        local["java"][args]()

        psql = local[str(PG_DIR / "./build/bin/psql")]

        if config["pg_stat_statements"]:
            with (results_dir / "stat_file.csv").open("w") as f:
                stats_result = psql[
                    "-d",
                    "benchbase",
                    "--csv",
                    "-c",
                    "SELECT * FROM pg_stat_statements;",
                ]()
                f.write(stats_result)

        if config["pg_store_plans"]:
            with (results_dir / "plan_file.csv").open("w") as f:
                plans_result = psql[
                    "-d",
                    "benchbase",
                    "--csv",
                    "-c",
                    "SELECT queryid, planid, plan FROM pg_store_plans ORDER BY queryid, planid;",
                ]()

                f.write(plans_result)

        # Move benchbase results to experiment results directory
        shutil.move(str(benchbase_snapshot_dir / "results"), benchbase_results_dir)
        time.sleep(10)  # Allow TScout Collector to finish getting results
    except (KeyboardInterrupt, FileNotFoundError, ProcessExecutionError) as err:
        cleanup(err, terminate=True, message="Error running Benchbase")


def exec_sqlsmith(bench_db: str) -> None:
    """Run SQLSmith queries over the specific benchmark database."""
    try:
        # Add SQLSmith user to benchbase DB with non-superuser privileges
        query(
            "benchbase", "CREATE ROLE sqlsmith WITH PASSWORD 'password' INHERIT LOGIN;"
        )

        for table in BENCHDB_TO_TABLES[bench_db]:
            logging.info("Granting SQLSmith permissions on table: %s", table)
            query(
                "benchbase",
                "CREATE ROLE sqlsmith WITH PASSWORD 'password' INHERIT LOGIN;",
            )
            query(
                "benchbase",
                f"GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO sqlsmith;",
            )

        logging.info("Begin SQLSmith Execution")
        os.chdir(SQLSMITH_DIR)
        (
            local["./sqlsmith"][
                '''--target="host=localhost port=5432 dbname=benchbase connect_timeout=10"''',
                "--seed=42",
                "--max-queries=10000",
                "--exclude-catalog",
            ]()
        )
    except (KeyboardInterrupt, FileNotFoundError, ProcessExecutionError) as err:
        cleanup(err, terminate=True, message="Error running SQLSmith")


def cleanup(err: Optional[BaseException], terminate: bool, message: str = "") -> None:
    """Clean up the TScout and Postgres processes after either a successful or failed run."""

    if len(message) > 0:
        logging.error(message)

    if err is not None:
        logging.error("Error: %s, %s", type(err), err)

    cleanup_script_path = Path(__file__).parent / "cleanup.py"
    sudo["python3"][cleanup_script_path]()
    time.sleep(2)  # Allow TScout poison pills to propagate

    # Exit the program if the caller requested it (only happens on error)
    if terminate:
        sys.exit(1)


def run(
    bench_db: str,
    results_dir: Path,
    benchbase_results_dir: Path,
    config: dict[str, Any],
) -> None:
    """Run an experiment"""
    assert results_dir.exists(), f"Results directory does not exist: {results_dir}"

    check_orphans()
    init_pg(config)
    init_benchbase(bench_db, benchbase_results_dir)

    # reload config to make a new logfile
    os.chdir(PG_DIR)
    pg_ctl = local["./build/bin/pg_ctl"]
    pg_ctl["stop", "-D", "data", "-m", "smart"]()

    # remove pre-existing logs
    for log_path in [
        fp for fp in (PG_DIR / "data/log").glob("*") if fp.suffix in ["csv", "log"]
    ]:
        log_path.unlink()

    pg_ctl["-D", "data", "-o", "-W 2", "start"] & FG

    if config["pg_prewarm"]:
        pg_prewarm(bench_db)

    if config["pg_analyze"]:
        pg_analyze(bench_db)

    init_tscout(results_dir)
    exec_benchbase(bench_db, results_dir, benchbase_results_dir, config)

    log_fps = list((PG_DIR / "data/log").glob("*.log"))
    assert (
        len(log_fps) == 1
    ), f"Expected 1 Postgres log file, found {len(log_fps)}, {log_fps}"
    shutil.move(str(log_fps[0]), str(results_dir))

    log_fps = list(results_dir.glob("*.log"))
    assert (
        len(log_fps) == 1
    ), f"Expected 1 Result log file, found {len(log_fps)}, {log_fps}"
    log_fps[0].rename(results_dir / "pg_log.log")

    logging.info("Finished benchmark run for %s", bench_db)

    cleanup(err=None, terminate=False, message="")


def main(config_name: str) -> None:
    config_path = CONFIG_DIR / f"{config_name}.yaml"
    with config_path.open("r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)["datagen"]
    init_logging(config["log_level"])

    # get sudo authentication for TScout
    sudo["pwd"]()

    # validate the benchmark databases from the config
    bench_dbs = config["bench_dbs"]
    for bench_db in bench_dbs:
        if bench_db not in BENCHDB_TO_TABLES:
            raise ValueError(f"Invalid benchmark database: {bench_db}")

    # Setup experiment directory
    experiment_name = f"experiment-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    modes = ["train", "eval"] if not config["debug"] else ["debug"]

    for mode in modes:
        mode_dir = BEHAVIOR_DATA_DIR / mode / experiment_name

        # copy datagen configuration to the output directory
        Path(mode_dir).mkdir(parents=True)
        shutil.copy(config_path, mode_dir)

        for bench_db in bench_dbs:
            results_dir = mode_dir / bench_db
            Path(results_dir).mkdir()
            benchbase_results_dir = results_dir / "benchbase"
            Path(benchbase_results_dir).mkdir(exist_ok=True)
            logging.info(
                "Running experiment: %s with bench_db: %s and results_dir: %s",
                experiment_name,
                bench_db,
                results_dir,
            )
            run(bench_db, results_dir, benchbase_results_dir, config)
