from pathlib import Path

from plumbum import local

import dodos.benchbase
import dodos.noisepage
from dodos import VERBOSITY_DEFAULT, default_artifacts_path, default_build_path

ARTIFACTS_PATH = default_artifacts_path()
BUILD_PATH = default_build_path()

# Input: various configuration files.
DATAGEN_CONFIG_FILE = Path("config/behavior/datagen.yaml").absolute()
MODELING_CONFIG_FILE = Path("config/behavior/modeling.yaml").absolute()
POSTGRESQL_CONF = Path("config/postgres/default_postgresql.conf").absolute()

# Scratch work.
BUILD_DATAGEN_PATH = BUILD_PATH / "datagen"

# Output: model directory.
ARTIFACT_WORKLOADS = ARTIFACTS_PATH / "workloads"
ARTIFACT_DATA_TRAIN = ARTIFACTS_PATH / "data/train"
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


def task_behavior_datagen():
    """
    Behavior modeling: generate training data and perform plan differencing.
    """
    # There should be no scenario in which you do NOT want plan differencing.
    # So it doesn't make sense to expose a separate task just for that.
    datagen_args = (
        f"--benchbase-user {dodos.benchbase.DEFAULT_USER} "
        f"--benchbase-pass {dodos.benchbase.DEFAULT_PASS} "
        f"--config-file {DATAGEN_CONFIG_FILE} "
        f"--dir-benchbase {dodos.benchbase.ARTIFACTS_PATH} "
        f"--dir-benchbase-config {dodos.benchbase.CONFIG_FILES} "
        f"--dir-noisepage-bin {dodos.noisepage.ARTIFACTS_PATH} "
        f"--dir-tscout {dodos.noisepage.BUILD_PATH / 'cmudb/tscout'} "
        f"--dir-output {ARTIFACT_DATA_TRAIN} "
        f"--dir-tmp {BUILD_DATAGEN_PATH} "
        f"--path-noisepage-conf {POSTGRESQL_CONF} "
        "--tscout-wait-sec 2 "
    )
    datadiff_args = f"--dir-datagen-data {ARTIFACT_DATA_TRAIN} --dir-output {ARTIFACT_DATA_DIFF} "

    return {
        "actions": [
            f"mkdir -p {ARTIFACT_DATA_TRAIN}",
            f"mkdir -p {ARTIFACT_DATA_DIFF}",
            f"mkdir -p {BUILD_DATAGEN_PATH}",
            f"python3 -m behavior datagen {datagen_args}",
            # Immediately perform plan differencing here since the models
            # suck without differencing anyway.
            f"python3 -m behavior datadiff {datadiff_args}",
        ],
        "file_dep": [
            dodos.benchbase.ARTIFACT_benchbase,
            dodos.noisepage.ARTIFACT_postgres,
        ],
        "targets": [ARTIFACT_DATA_TRAIN, ARTIFACT_DATA_DIFF],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_behavior_train():
    """
    Behavior modeling: train OU models.
    """
    train_args = (
        f"--config-file {MODELING_CONFIG_FILE} "
        f"--dir-data-train {ARTIFACT_DATA_DIFF / 'diff/train'} "
        f"--dir-data-eval {ARTIFACT_DATA_DIFF / 'diff/eval'} "
        f"--dir-output {ARTIFACT_MODELS} "
    )

    return {
        "actions": [
            f"mkdir -p {ARTIFACT_MODELS}",
            f"python3 -m behavior train {train_args}",
        ],
        "targets": [ARTIFACT_MODELS],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_behavior_microservice():
    """
    Behavior modeling: models as a microservice (via Flask).
    """

    def run_microservice():
        # Find the latest experiment.
        # TODO(WAN): Make this configurable.
        experiment_list = sorted(exp_path for exp_path in ARTIFACT_MODELS.glob("*"))
        assert len(experiment_list) > 0, "No experiments found."
        experiment_name = experiment_list[-1]

        server_cmd = local["python3"]["-m", "behavior", "microservice", "--models-path", experiment_name]
        dest_stdout = "artifacts/behavior/microservice/microservice.out"
        ret = server_cmd.run_nohup(stdout=dest_stdout)
        print(f"Behavior models microservice: {dest_stdout} {ret.pid}")

    return {
        "actions": ["mkdir -p ./artifacts/behavior/microservice/", run_microservice],
        "verbosity": VERBOSITY_DEFAULT,
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
