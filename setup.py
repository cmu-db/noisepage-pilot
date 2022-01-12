from setuptools import find_packages, setup

setup(
    name="noisepage_pilot",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "black",
        "doit",
        "flake8",
        "isort",
        "lightgbm",
        "mypy",
        "numpy",
        "pandas",
        "plumbum",
        "psycopg",
        "psutil",
        "pydotplus",
        "pylint",
        "pyyaml",
        "scikit-learn",
        "toml",
        "tqdm",
        "types-PyYAML",
    ],
    python_requires=">=3.8",
)
