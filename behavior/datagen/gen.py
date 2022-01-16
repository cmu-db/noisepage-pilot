from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil
import yaml
from plumbum import ProcessExecutionError, cli, cmd, local

from behavior import BENCHDB_TO_TABLES

# TODO(WAN): This entire file can probably be replaced with doit somehow.

logger = logging.getLogger(__name__)


class DataGeneratorCLI(cli.Application):
    config_file = cli.SwitchAttr(
        "--config-file",
        Path,
        mandatory=True,
        help="Path to configuration YAML containing datagen parameters.",
    )
    dir_noisepage_bin = cli.SwitchAttr(
        "--dir-noisepage-bin",
        Path,
        mandatory=True,
        help="Directory containing NoisePage binaries.",
    )
    dir_benchbase = cli.SwitchAttr(
        "--dir-benchbase",
        Path,
        mandatory=True,
        help="Path to extracted BenchBase.",
    )
    dir_benchbase_config = cli.SwitchAttr(
        "--dir-benchbase-config",
        Path,
        mandatory=True,
        help="Path to BenchBase config files.",
    )
    dir_tscout = cli.SwitchAttr(
        "--dir-tscout",
        Path,
        mandatory=True,
        help="Path to TScout.",
    )
    dir_output = cli.SwitchAttr(
        "--dir-output",
        Path,
        mandatory=True,
        help="Directory to write generated data to.",
    )
    dir_tmp = cli.SwitchAttr(
        "--dir-tmp",
        Path,
        mandatory=True,
        help="Directory for temporary files created by the datagen process.",
    )
    path_noisepage_conf = cli.SwitchAttr(
        "--path-noisepage-conf",
        Path,
        mandatory=True,
        help="Path to the postgresql.conf for NoisePage.",
    )
    tscout_wait_sec = cli.SwitchAttr(
        "--tscout-wait-sec",
        int,
        default=2,
        help="Argument to postgres -W, i.e., delay (s) after starting server process.",
    )
    benchbase_user = cli.SwitchAttr(
        "--benchbase-user",
        str,
        default="admin",
        help="Username to create and use for BenchBase.",
    )
    benchbase_pass = cli.SwitchAttr(
        "--benchbase-pass",
        str,
        default="password",
        help="Password to create and use for BenchBase.",
    )

    def query(self, db, query_str, psql_args=None):
        """
        Convenience wrapper around psql.

        Parameters
        ----------
        db : str
            The name of the database.
        query_str : str
            The query string to execute.
        psql_args : List[str]
            Any additional arguments to psql.
        """
        args = ["--dbname", db, "--command", query_str]
        if psql_args is not None:
            args.extend(psql_args)
        self.psql[args]()

    def _noisepage_init(self):
        """
        Initialize and run NoisePage.
        """
        try:
            # Delete the old PGDATA folder if it exists.
            pg_data_dir = self.dir_tmp_pg_data
            if pg_data_dir.exists():
                logger.debug("Removing existing PGDATA directory: %s", pg_data_dir)
                shutil.rmtree(pg_data_dir)

            # Initialize the PGDATA folder.
            logger.debug("PGDATA: Initializing.")
            self.initdb["--pgdata", pg_data_dir]()
            shutil.copy(self.path_noisepage_conf, pg_data_dir / "postgresql.conf")
            logger.debug("PGDATA: Initialized.")
            logger.debug("NoisePage: Starting.")
            self.pg_ctl[
                "start",
                "-D",
                pg_data_dir,
                "-o",
                f"-W {self.tscout_wait_sec}",
            ].run_fg()
            # TODO(WAN): When the above code is not run in the foreground,
            #  it occasionally hangs. ???
            logger.debug("NoisePage: Started.")

            # Initialize databases and create an admin user for BenchBase to use.
            logger.debug("NoisePage: Database and user setup.")
            self.createdb["test"]()
            self.query(
                "test",
                "CREATE ROLE admin WITH PASSWORD 'password' SUPERUSER CREATEDB CREATEROLE INHERIT LOGIN;",
            )
            self.createdb["--owner", "admin", "benchbase"]()

            # Enable capturing query IDs.
            logger.debug("NoisePage: Query ID setup.")
            self.query("test", "ALTER DATABASE test SET compute_query_id = 'ON';")
            self.query("benchbase", "ALTER DATABASE benchbase SET compute_query_id = 'ON';")

            # Set up extensions.
            logger.debug("NoisePage: Extension setup.")
            if self.config["auto_explain"]:
                logger.debug("NoisePage: auto_explain.")
                self.query("benchbase", "ALTER SYSTEM SET auto_explain.log_min_duration = 0;")
                self.pg_ctl["reload", "-D", pg_data_dir]()

            if self.config["pg_prewarm"]:
                logger.debug("NoisePage: pg_prewarm.")
                self.query("benchbase", "CREATE EXTENSION pg_prewarm;")

            if self.config["pg_stat_statements"]:
                logger.debug("NoisePage: pg_stat_statements.")
                self.query("benchbase", "CREATE EXTENSION pg_stat_statements;")

            if self.config["pg_store_plans"]:
                logger.debug("NoisePage: pg_store_plans.")
                self.query("benchbase", "CREATE EXTENSION pg_store_plans;")

            # Turn off pager to avoid needing tty interaction.
            self.query("benchbase", "SELECT 1;", ["--pset", "pager=off"])

        except (KeyboardInterrupt, FileNotFoundError, ProcessExecutionError) as err:
            self.clean(err, terminate=True, message="Error initializing NoisePage.")

    def _benchbase_init(self, benchmark, benchbase_results_dir):
        """
        Initialize BenchBase (create and load).

        Parameters
        ----------
        benchmark : str
            The BenchBase benchmark to be created and loaded.
        benchbase_results_dir : Path
            Directory for Benchbase results.
        """
        try:
            # Copy the BenchBase config to the output directory.
            input_cfg_path = self.dir_benchbase_config / f"{benchmark}_config.xml"
            shutil.copy(input_cfg_path, benchbase_results_dir)
            cfg_path = (benchbase_results_dir / f"{benchmark}_config.xml").absolute()

            old_wd = os.getcwd()
            os.chdir(self.dir_benchbase)
            logger.debug("BenchBase starting create and load: %s", benchmark)
            benchbase_cmd = [
                "-jar",
                "benchbase.jar",
                "--bench",
                benchmark,
                "--config",
                cfg_path,
                "--create=true",
                "--load=true",
                "--execute=false",
            ]
            local["java"][benchbase_cmd]()
            logger.debug("BenchBase completed create and load: %s", benchmark)
            os.chdir(old_wd)
        except (KeyboardInterrupt, FileNotFoundError, ProcessExecutionError) as err:
            self.clean(err, terminate=True, message="Error initializing BenchBase.")

    def _benchbase_exec(self, benchmark, benchbase_results_dir):
        """
        Run BenchBase (execute).

        Parameters
        ----------
        benchmark : str
            The BenchBase benchmark to be run.
        benchbase_results_dir : Path
            Directory for Benchbase results.
        """
        try:
            if self.config["pg_stat_statements"]:
                self.query("benchbase", "SELECT pg_stat_statements_reset();")

            if self.config["pg_store_plans"]:
                self.query("benchbase", "SELECT pg_store_plans_reset();")

            cfg_path = (benchbase_results_dir / f"{benchmark}_config.xml").absolute()

            old_wd = os.getcwd()
            os.chdir(self.dir_benchbase)
            logger.debug("BenchBase starting execute: %s", benchmark)
            args = [
                "-jar",
                "benchbase.jar",
                "--bench",
                benchmark,
                "--config",
                cfg_path,
                "--create=false",
                "--load=false",
                "--execute=true",
            ]
            local["java"][args]()
            logger.debug("BenchBase completed execute: %s", benchmark)
            os.chdir(old_wd)

            if self.config["pg_stat_statements"]:
                with (self.dir_output / "stat_file.csv").open("w") as f:
                    stats_result = self.psql[
                        "--dbname",
                        "benchbase",
                        "--csv",
                        "--command",
                        "SELECT * FROM pg_stat_statements;",
                    ]()
                    f.write(stats_result)

            if self.config["pg_store_plans"]:
                with (self.dir_output / "plan_file.csv").open("w") as f:
                    plans_result = self.psql[
                        "--dbname",
                        "benchbase",
                        "--csv",
                        "--command",
                        "SELECT queryid, planid, plan FROM pg_store_plans ORDER BY queryid, planid;",
                    ]()
                    f.write(plans_result)

            # Move BenchBase results to experiment results directory.
            shutil.move(str(self.dir_benchbase / "results"), benchbase_results_dir)
        except (KeyboardInterrupt, FileNotFoundError, ProcessExecutionError) as err:
            self.clean(err, terminate=True, message="Error running BenchBase.")

    @staticmethod
    def exec_sqlsmith(benchmark):
        """Run SQLSmith queries over the specific benchmark database."""
        raise Exception("Wan needs to fix this.")
        # try:
        #     # Add SQLSmith user to benchbase DB with non-superuser privileges
        #     self.query(
        #         "benchbase",
        #         "CREATE ROLE sqlsmith WITH PASSWORD 'password' INHERIT LOGIN;",
        #     )
        #
        #     for table in BENCHDB_TO_TABLES[benchmark]:
        #         logger.info("Granting SQLSmith permissions on table: %s", table)
        #         self.query(
        #             "benchbase",
        #             "CREATE ROLE sqlsmith WITH PASSWORD 'password' INHERIT LOGIN;",
        #         )
        #         self.query(
        #             "benchbase",
        #             f"GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO sqlsmith;",
        #         )
        #
        #     logger.info("Begin SQLSmith Execution")
        #     # os.chdir(SQLSMITH_DIR)
        #     (
        #         local["./sqlsmith"][
        #             '''--target="host=localhost port=5432 dbname=benchbase connect_timeout=10"''',
        #             "--seed=42",
        #             "--max-queries=10000",
        #             "--exclude-catalog",
        #         ]()
        #     )
        # except (KeyboardInterrupt, FileNotFoundError, ProcessExecutionError) as err:
        #     self.clean(err, terminate=True, message="Error running SQLSmith.")

    @staticmethod
    def _assert_no_orphans():
        """
        Assert that no TScout or NoisePage processes exist.
        """

        tscout_process_names = {
            "TScout Coordinator",
            "TScout Processor",
            "TScout Collector",
        }
        pg_procs, tscout_procs = [], []
        for proc in psutil.process_iter(["pid", "name", "username", "ppid", "create_time"]):
            proc_name = proc.info["name"].lower()
            if "postgres" in proc_name:
                pg_procs.append(proc)
            if not tscout_process_names.isdisjoint(proc.info["name"]):
                tscout_procs.append(proc)

        assert len(pg_procs) == 0, f"Found orphaned postgres: {pg_procs}"
        assert len(tscout_procs) == 0, f"Found orphaned tscout: {tscout_procs}"

    def pg_analyze(self, benchmark):
        """
        Run ANALYZE on all the tables in the given benchmark.
        This updates internal statistics for estimating cardinalities and costs.

        Parameters
        ----------
        benchmark : str
            The benchmark whose tables should be analyzed.
        """
        try:
            logger.debug("Running ANALYZE.")
            for table in BENCHDB_TO_TABLES[benchmark]:
                self.query("benchbase", f"ANALYZE VERBOSE {table};")
        except ProcessExecutionError as err:
            self.clean(err, terminate=True, message="Error during analyze.")

    def pg_prewarm(self, benchmark):
        """
        Run pg_prewarm() on all the tables in the given benchmark.
        This warms the buffer pool and OS page cache.

        Parameters
        ----------
        benchmark : str
            The benchmark whose tables should be prewarmed.
        """
        try:
            logger.debug("Running prewarm.")
            for table in BENCHDB_TO_TABLES[benchmark]:
                self.query("benchbase", f"SELECT * from pg_prewarm('{table}');")
        except (KeyboardInterrupt, FileNotFoundError, ProcessExecutionError) as err:
            self.clean(err, terminate=True, message="Error prewarming NoisePage.")

    def _tscout_init(self, output_dir):
        """
        Initialize and launch TScout.

        Parameters
        ----------
        output_dir : Path
            Directory for output files.
        """
        try:
            # This assumes that the oldest postgres PID is the postmaster.
            postmaster_pid = local["pgrep"]["-ox", "postgres"]()
            dir_tscout_output = output_dir / "tscout"
            dir_tscout_output.mkdir(parents=True, exist_ok=True)
            dir_tscout_output = dir_tscout_output.absolute()
            logger.debug("Attaching TScout.")
            old_wd = os.getcwd()
            os.chdir(self.dir_tscout)
            cmd.sudo["python3"]["tscout.py", postmaster_pid, "--outdir", dir_tscout_output].run_bg(
                stdout=sys.stdout, stderr=sys.stderr
            )
            os.chdir(old_wd)
        except (FileNotFoundError, ProcessExecutionError) as err:
            self.clean(err, terminate=True, message="Error initializing TScout.")

        # TODO(GARRISON/WAN): This is a hack.
        #   Allow TScout to attach before BenchBase execution begins.
        time.sleep(5)

    def clean(self, err=None, terminate=False, message=""):
        """
        Clean up the TScout and NoisePage processes.

        Parameters
        ----------
        err : Exception | None
            The exception, if any.
        terminate : bool
            True if the program should terminate.
        message : str
            Additional message to be logged.
        """

        # Perform any necessary logging.
        if len(message) > 0:
            logger.error(message)
        if err is not None:
            logger.error("Error: %s, %s", type(err), err)

        # Kill TScout and NoisePage.
        # retcode=None means don't validate return code.
        cmd.sudo["pkill", "-i", "postgres"](retcode=None)
        cmd.sudo["pkill", "-i", "tscout"](retcode=None)

        # Because TScout has to execute with sudo,
        # the results are owned by root.
        # Take ownership of TScout's results.
        # TODO(GARRISON/WAN):
        #   TScout takes some time to clean up and write the final result files.
        time.sleep(5)
        owner = Path(__file__).owner()
        training_data_dir = self.dir_output
        if training_data_dir.exists():
            logger.debug("Taking ownership of: %s", training_data_dir)
            cmd.sudo["chown", "--recursive", owner, training_data_dir]()
        # Nuke that irritating pycache folder.
        cmd.sudo["rm", "-rf", self.dir_tscout / "__pycache__"]()

        # Exit if necessary, e.g., exceptional behavior occurred.
        if terminate:
            sys.exit(1)

    def run_experiment(self, benchmark, output_dir, benchbase_results_dir):
        """
        Run the specified experiment.

        Parameters
        ----------
        benchmark : str
            The BenchBase benchmark to run.
        output_dir : Path
            Directory for output files.
        benchbase_results_dir : Path
            Directory for BenchBase results.
        """
        assert output_dir.exists(), f"Output directory does not exist: {output_dir}"

        self._assert_no_orphans()
        self._noisepage_init()
        self._benchbase_init(benchmark, benchbase_results_dir)

        # Reload the configuration to make a new logfile.
        self.pg_ctl["stop", "-D", self.dir_tmp_pg_data, "-m", "smart"].run_fg()

        # Remove existing logfiles, if any.
        for log_path in [path for path in (self.dir_tmp_pg_data / "log").glob("*") if path.suffix in ["csv", "log"]]:
            log_path.unlink()

        # Start NoisePage.
        self.pg_ctl["start", "-D", self.dir_tmp_pg_data, "-o", "-W 2"].run_fg()

        # Run prewarm so that disk isn't a factor.
        if self.config["pg_prewarm"]:
            self.pg_prewarm(benchmark)

        # Run analyze so that table stats are up to date.
        if self.config["pg_analyze"]:
            self.pg_analyze(benchmark)

        # Attach TScout.
        self._tscout_init(output_dir)
        # Execute BenchBase.
        self._benchbase_exec(benchmark, benchbase_results_dir)
        # TODO(GARRISON/WAN): This is a hack.
        #   Allow TScout Collector to finish getting results.
        time.sleep(10)

        log_fps = list((self.dir_tmp_pg_data / "log").glob("*.log"))
        assert len(log_fps) == 1, f"Expected 1 log file, found {len(log_fps)}, {log_fps}"
        shutil.move(str(log_fps[0]), str(self.dir_output / "pg_log.log"))

        self.clean(err=None, terminate=False, message="")
        logger.debug("Finished benchmark run: %s", benchmark)

    def main(self):
        # Bind NoisePage binaries.
        self.psql = local[str(self.dir_noisepage_bin / "psql")]
        self.pg_ctl = local[str(self.dir_noisepage_bin / "pg_ctl")]
        self.postgres = local[str(self.dir_noisepage_bin / "postgres")]
        self.createdb = local[str(self.dir_noisepage_bin / "createdb")]
        self.initdb = local[str(self.dir_noisepage_bin / "initdb")]

        # Bind any working folders.
        self.dir_tmp_pg_data = self.dir_tmp / "pgdata"

        # Load the config file.
        config_path = Path(self.config_file)
        with config_path.open("r", encoding="utf-8") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)["datagen"]
        logger.setLevel(self.config["log_level"])
        # Get sudo authentication for TScout.
        cmd.sudo["--validate"]()

        # Validate the chosen benchmarks from the config.
        benchmarks = self.config["benchmarks"]
        for benchmark in benchmarks:
            if benchmark not in BENCHDB_TO_TABLES:
                raise ValueError(f"Invalid benchmark: {benchmark}")

        # Setup experiment directory.
        experiment_name = f"experiment-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        modes = ["train", "eval"] if not self.config["debug"] else ["debug"]

        # For each training mode, ...
        for mode in modes:
            # Create the output directory.
            mode_dir = Path(self.dir_output) / mode / experiment_name
            Path(mode_dir).mkdir(parents=True)
            # Copy the config file to the output directory.
            shutil.copy(config_path, mode_dir)

            # For each benchmark, ...
            for benchmark in benchmarks:
                # Run BenchBase to generate training data.
                results_dir = Path(mode_dir / benchmark)
                results_dir.mkdir()
                benchbase_results_dir = Path(results_dir / "benchbase")
                benchbase_results_dir.mkdir(exist_ok=True)
                logger.info(
                    "Running experiment %s with benchmark %s and results %s",
                    experiment_name,
                    benchmark,
                    results_dir,
                )
                self.run_experiment(benchmark, results_dir, benchbase_results_dir)


if __name__ == "__main__":
    DataGeneratorCLI.run()
