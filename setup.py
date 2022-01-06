from setuptools import find_packages, setup

setup(
    name="noisepage_pilot",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "psutil",
        "pyyaml",
        "pandas",
        "sklearn>=1.0.0",
        "pydotplus",
        "lightgbm",
        "dataclasses",
        "isort",
        "black",
        "flake8",
        "pylint",
        "mypy",
        "plumbum",
        "doit",
        "psycopg",
        "toml",
    ],
    python_requires=">=3.9",
)
