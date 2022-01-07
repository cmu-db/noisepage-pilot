from __future__ import annotations

import argparse
import logging

from behavior.datagen import gen
from behavior.modeling import train
from behavior.plans import diff
from behavior.util import get_latest_experiment, init_logging

parser = argparse.ArgumentParser(
    description="Run an experiment with Postgres, Benchbase, and TScout"
)
parser.add_argument("--config", type=str, default="default")
parser.add_argument("--datagen", action="store_true")
parser.add_argument("--diff", action="store_true")
parser.add_argument("--train", action="store_true")
args = parser.parse_args()
config_name = args.config
run_datagen = args.datagen
run_diff = args.diff
run_train = args.train

init_logging("INFO")
logging.info(
    "Behavior Modeling Configuration: Datagen: %s | Diff: %s | Train: %s",
    run_datagen,
    run_diff,
    run_train,
)

if run_datagen:
    gen.main(config_name)

latest_experiment = get_latest_experiment()

if run_diff:
    diff.main(latest_experiment)

if run_train:
    train.main("default")
