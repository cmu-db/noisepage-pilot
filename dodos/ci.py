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
            # We currently exclude behavior/modeling/featurewiz (which is a fork from AutoViML/featurewiz)
            # from going through black/isort/flake8. This is done to prevent any merge conflicts when trying
            # to pull future upstream changes from AutoViML.
            *[
                f"black --exclude behavior/modeling/featurewiz {config['check']} --verbose {folder}"
                for folder in folders
            ],
            *[f"isort --skip behavior/modeling/featurewiz {config['check']} {folder}" for folder in folders],
            *[f"flake8 --exclude behavior/modeling/featurewiz --statistics {folder}" for folder in folders],
            # TODO(WAN): Only run pylint on behavior for now.
            *[f"pylint --verbose {folder}" for folder in ["behavior"]],
        ],
        "verbosity": VERBOSITY_DEFAULT,
        "uptodate": [False],
    }
