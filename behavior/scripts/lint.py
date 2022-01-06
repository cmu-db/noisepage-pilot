#!/usr/bin/env python

from pathlib import Path

from plumbum import FG
from plumbum.cmd import black, flake8, isort, mypy, pylint

root = str(Path.home() / "noisepage-pilot" / "behavior")

# We don't need retcode validation for local use, but it will
# be useful for CI
black[root, "--exclude=.ipynb"] & FG(retcode=None)
isort[root] & FG(retcode=None)
flake8[root] & FG(retcode=None)
mypy[root] & FG(retcode=None)
pylint[f"{root}/src"] & FG(retcode=None)
