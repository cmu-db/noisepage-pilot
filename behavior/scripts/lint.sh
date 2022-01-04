#!/bin/bash

black . --exclude=\.ipynb
isort .
flake8 .
mypy .
pylint src
