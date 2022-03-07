import os
from pathlib import Path

from doit.action import CmdAction
from plumbum import local

import dodos.benchbase
import dodos.noisepage
from behavior import BENCHDB_TO_TABLES
from dodos import VERBOSITY_DEFAULT, default_artifacts_path, default_build_path
from dodos.benchbase import ARTIFACTS_PATH as BENCHBASE_ARTIFACTS_PATH
from dodos.noisepage import (
    ARTIFACTS_PATH as NOISEPAGE_ARTIFACTS_PATH,
    ARTIFACT_pgdata,
    ARTIFACT_psql,
)

ARTIFACTS_PATH = default_artifacts_path()
BUILD_PATH = default_build_path()

# Input: various configuration files.
DATAGEN_CONFIG_FILE = Path("config/behavior/datagen.yaml").absolute()
MODELING_CONFIG_FILE = Path("config/behavior/modeling.yaml").absolute()
POSTGRESQL_CONF = Path("config/postgres/default_postgresql.conf").absolute()

# Output: model directory.
ARTIFACT_WORKLOADS = ARTIFACTS_PATH / "workloads"
ARTIFACT_DATA_RAW = ARTIFACTS_PATH / "data/raw"
ARTIFACT_DATA_DIFF = ARTIFACTS_PATH / "data/diff"
ARTIFACT_MODELS = ARTIFACTS_PATH / "models"


def task_behavior_generate_workloads():
    """
    Behavior modeling: generate the workloads that we plan to execute for training data.
    """
    generate_workloads_args = (
        f"--config-file {DATAGEN_CONFIG_FILE} "
        f"--postgresql-config-file {POSTGRESQL_CONF} "
        f"--dir-benchbase-config {dodos.benchbase.CONFIG_FILES} "
        f"--dir-output {ARTIFACT_WORKLOADS} "
    )

    def conditional_clear(clear_existing):
        if clear_existing != "False":
            local["rm"]["-rf"][f"{ARTIFACT_WORKLOADS}"].run()

        return None

    return {
        "actions": [
            conditional_clear,
            f"python3 -m behavior generate_workloads {generate_workloads_args}",
        ],
        "file_dep": [
            dodos.benchbase.ARTIFACT_benchbase,
            dodos.noisepage.ARTIFACT_postgres,
        ],
        "targets": [ARTIFACT_WORKLOADS],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "clear_existing",
                "long": "clear_existing",
                "help": "Remove existing generated workloads.",
                "default": True,
            },
        ],
    }


def task_behavior_execute_workloads():
    """
    Behavior modeling: execute workloads to generate training data.
    """
    execute_args = (
        f"--workloads={ARTIFACT_WORKLOADS} "
        f"--output-dir={ARTIFACT_DATA_RAW} "
        f"--pgdata={ARTIFACT_pgdata} "
        f"--benchbase={BENCHBASE_ARTIFACTS_PATH} "
        f"--pg_binaries={NOISEPAGE_ARTIFACTS_PATH} "
    )

    return {
        "actions": [
            f"mkdir -p {ARTIFACT_DATA_RAW}",
            f"behavior/datagen/run_workloads.sh {execute_args}",
        ],
        "file_dep": [
            dodos.benchbase.ARTIFACT_benchbase,
            dodos.noisepage.ARTIFACT_postgres,
            dodos.noisepage.ARTIFACT_pg_ctl,
        ],
        "targets": [ARTIFACT_DATA_RAW],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_behavior_perform_plan_diff():
    """
    Behavior modeling: perform plan differencing.
    """

    def datadiff_action(glob_pattern):
        datadiff_args = f"--dir-datagen-data {ARTIFACT_DATA_RAW} " f"--dir-output {ARTIFACT_DATA_DIFF} "

        if glob_pattern is not None:
            datadiff_args = datadiff_args + f"--glob-pattern '{glob_pattern}'"

        # Include the cython compiled modules in PYTHONPATH.
        return f"PYTHONPATH=artifacts/:$PYTHONPATH python3 -m behavior datadiff {datadiff_args}"

    return {
        "actions": [
            # The following command is necessary to force a rebuild everytime. Recompile diff_c.pyx.
            "rm -f behavior/plans/diff_c.c",
            f"python3 behavior/plans/setup.py build_ext --build-lib artifacts/ --build-temp {default_build_path()}",
            f"mkdir -p {ARTIFACT_DATA_DIFF}",
            CmdAction(datadiff_action, buffering=1),
        ],
        "file_dep": [
            dodos.benchbase.ARTIFACT_benchbase,
            dodos.noisepage.ARTIFACT_postgres,
        ],
        "targets": [ARTIFACT_DATA_DIFF],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "glob_pattern",
                "long": "glob_pattern",
                "help": "Glob pattern for selecting which experiments to perform differencing.",
                "default": None,
            },
        ],
    }


def task_behavior_train():
    """
    Behavior modeling: train OU models.
    """

    def train_cmd(train_experiment_names, train_benchmark_names, eval_experiment_names, eval_benchmark_names):
        train_args = (
            f"--config-file {MODELING_CONFIG_FILE} "
            f"--dir-data {ARTIFACT_DATA_DIFF} "
            f"--dir-output {ARTIFACT_MODELS} "
        )

        args = {
            "--train-experiment-names": train_experiment_names,
            "--train-benchmark-names": train_benchmark_names,
            "--eval-experiment-names": eval_experiment_names,
            "--eval-benchmark-names": eval_benchmark_names,
        }

        for k, v in args.items():
            if v is not None:
                train_args = train_args + f"{k}='{v}' "

        return f"python3 -m behavior train {train_args}"

    return {
        "actions": [f"mkdir -p {ARTIFACT_MODELS}", CmdAction(train_cmd)],
        "targets": [ARTIFACT_MODELS],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "train_experiment_names",
                "long": "train_experiment_names",
                "help": "Comma separated experiments/experiments glob patterns for training models.",
                "default": None,
            },
            {
                "name": "train_benchmark_names",
                "long": "train_benchmark_names",
                "help": "Comma separated benchmarks/benchmarks glob patterns for training models.",
                "default": "*",
            },
            {
                "name": "eval_experiment_names",
                "long": "eval_experiment_names",
                "help": "Comma separated experiments/experiments glob patterns for evaluating models on.",
                "default": None,
            },
            {
                "name": "eval_benchmark_names",
                "long": "eval_benchmark_names",
                "help": "Comma separated benchmarks/benchmarks glob patterns for evaluating models on.",
                "default": "*",
            },
        ],
    }


def task_behavior_microservice():
    """
    Behavior modeling: models as a microservice (via Flask).
    """

    def run_microservice(models):
        if models is None:
            # Find the latest experiment by last modified timestamp.
            experiment_list = sorted((exp_path for exp_path in ARTIFACT_MODELS.glob("*")), key=os.path.getmtime)
            assert len(experiment_list) > 0, "No experiments found."
            models = experiment_list[-1]
        else:
            assert os.path.isdir(models), f"Specified path {models} is not a valid directory."

        server_cmd = local["python3"]["-m", "behavior", "microservice", "--models-path", models]
        dest_stdout = "artifacts/behavior/microservice/microservice.out"
        ret = server_cmd.run_nohup(stdout=dest_stdout)
        print(f"Behavior models microservice: {dest_stdout} {ret.pid}")

    return {
        "actions": ["mkdir -p ./artifacts/behavior/microservice/", run_microservice],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "models",
                "long": "models",
                "help": "Path to folder containing models that should be used.",
                "default": None,
            },
        ],
    }


def task_behavior_microservice_kill():
    """
    Behavior modeling: kill all running microservices.
    """
    return {
        "actions": [
            "pkill --full '^Behavior Models Microservice' || true",
        ],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_behavior_pg_analyze_benchmark():
    """
    Behavior modeling:

    Run ANALYZE on all the tables in the given benchmark.
    This updates internal statistics for estimating cardinalities and costs.

    Parameters
    ----------
    benchmark : str
        The benchmark whose tables should be analyzed.
    """

    def pg_analyze(benchmark):
        if benchmark is None or benchmark not in BENCHDB_TO_TABLES:
            print(f"Benchmark {benchmark} is not specified or does not exist.")
            return False

        for table in BENCHDB_TO_TABLES[benchmark]:
            query = f"ANALYZE VERBOSE {table};"
            local[str(ARTIFACT_psql)]["--dbname=benchbase"]["--command"][query]()

    return {
        "actions": [pg_analyze],
        "file_dep": [ARTIFACT_psql],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "benchmark",
                "long": "benchmark",
                "help": "Benchmark whose tables should be analyzed.",
                "default": None,
            },
        ],
    }


def task_behavior_pg_prewarm_benchmark():
    """
    Behavior modeling:

    Run pg_prewarm() on all the tables in the given benchmark.
    This warms the buffer pool and OS page cache.

    Parameters
    ----------
    benchmark : str
        The benchmark whose tables should be prewarmed.
    """

    def pg_prewarm(benchmark):
        if benchmark is None or benchmark not in BENCHDB_TO_TABLES:
            print(f"Benchmark {benchmark} is not specified or does not exist.")
            return False

        for table in BENCHDB_TO_TABLES[benchmark]:
            query = f"SELECT * FROM pg_prewarm('{table}');"
            local[str(ARTIFACT_psql)]["--dbname=benchbase"]["--command"][query]()

    return {
        "actions": [pg_prewarm],
        "file_dep": [ARTIFACT_psql],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "benchmark",
                "long": "benchmark",
                "help": "Benchmark whose tables should be analyzed.",
                "default": None,
            },
        ],
    }
