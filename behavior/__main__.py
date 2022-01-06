import argparse

from behavior import TRAIN_DATA_DIR
from behavior.datagen import generator
from behavior.modeling import train
from behavior.plans import diff

parser = argparse.ArgumentParser(description="Run an experiment with Postgres, Benchbase, and TScout")
parser.add_argument("--config", type=str, default="default")
parser.add_argument("--datagen", action="store_true")
parser.add_argument("--diff", action="store_true")
parser.add_argument("--train", action="store_true")
args = parser.parse_args()
config_name = args.config

if args.datagen:
    generator.main(config_name)

# get latest experiment and run differencing
experiment_list: list[str] = sorted([exp_path.name for exp_path in TRAIN_DATA_DIR.glob("*")])
if len(experiment_list) == 0:
    raise ValueError("No experiments found")

latest_experiment: str = experiment_list[-1]

if args.diff:
    diff.main(latest_experiment)


if args.train:
    train.main("default")
