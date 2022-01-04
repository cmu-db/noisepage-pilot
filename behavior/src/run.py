import argparse

from src import TRAIN_DATA_DIR
from src.datagen import datagen
from src.modeling import train
from src.plans import diff

parser = argparse.ArgumentParser(description="Run an experiment with Postgres, Benchbase, and TScout")
parser.add_argument("--config", type=str, default="default")
args = parser.parse_args()
config_name = args.config

# datagen.main(config_name)

# get latest experiment and run differencing
experiment_list: list[str] = sorted([exp_path.name for exp_path in TRAIN_DATA_DIR.glob("*")])
if len(experiment_list) == 0:
    raise ValueError("No experiments found")

latest_experiment: str = experiment_list[-1]
diff.main(latest_experiment)

# train.main(latest_experiment)
