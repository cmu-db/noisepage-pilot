from __future__ import annotations

import logging

from behavior import TRAIN_DATA_DIR


def get_latest_experiment() -> str:
    # get latest experiment and run differencing
    experiment_list: list[str] = sorted(
        [exp_path.name for exp_path in TRAIN_DATA_DIR.glob("*")]
    )
    if len(experiment_list) == 0:
        raise ValueError("No experiments found")

    latest_experiment: str = experiment_list[-1]
    return latest_experiment


def init_logging(level: str) -> None:
    if level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        raise ValueError(f"Invalid log level: {level}.")
    logging.basicConfig(format="%(levelname)s:%(asctime)s %(message)s", level=level)
