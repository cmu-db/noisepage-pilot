from doit import get_var

from dodos import VERBOSITY_DEFAULT


def task_ci_clean_slate():
    """
    CI: clear out all artifacts and build folders.
    """
    folders = ["artifacts", "build"]

    return {
        "actions": [
            *[f"sudo rm -rf {folder}" for folder in folders],
        ],
        "verbosity": VERBOSITY_DEFAULT,
        "uptodate": [False],
    }


def task_ci_python():
    """
    CI: this should be run and all warnings fixed before pushing commits.
    """
    folders = ["action", "behavior", "dodos", "forecast", "pilot"]
    config = {"check": "--check" if str(get_var("check")).lower() == "true" else ""}

    return {
        "actions": [
            *[f"black {config['check']} --verbose {folder}" for folder in folders],
            *[f"isort {config['check']} {folder}" for folder in folders],
            *[f"flake8 --statistics {folder}" for folder in folders],
            # TODO(WAN): Only run pylint on behavior for now.
            *[f"pylint --verbose {folder}" for folder in ["behavior"]],
        ],
        "verbosity": VERBOSITY_DEFAULT,
        "uptodate": [False],
    }
