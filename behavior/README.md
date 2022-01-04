# Behavior Modeling

This document details the core components of behavior modeling and how to use them.

# Pipeline

This diagram details the general workflow.

![Behavior Modeling Diagram](./docs/behavior_modeling_pipeline.svg)

# Key Terms

This diagram presents various key terminology relevant to the behavior modeling process.

![Key Terms](./docs/behavior_modeling_keyterms.svg)

## Data Generator (`datagen.py`)



## Training Data

- Separated into training and evaluation data in `behavior_modeling/data/train` and `behavior_modeling/data/evaluate`
- This is done because it makes data management simpler, and we can always easily add more training data.
- The alternative is to have the data stored jointly and maintain indexes or nested directories partitioning training and evaluation data.  This is typically only worth doing if training data is scarce and expensive because it allows for more flexible experimentation with minimal data.  This isn't a concern here, so we fully separate train and evaluation datasets.  This also avoids issues of intra-run data leakage; i.e. the data within a given round not being I.I.D.
- In the future, it may be best to store this data in a database because it is tabular and SQL-based access will allow for more flexible and dynamic construction of training/evaluation sets.  This was not a concern in MB2 because the data was statically created one time by handwritten microbenchmarks and all columns for all OUs were identical.


## Training (`train.py`)

Trains, evaluates, and serializes models, saving all results to `behavior_modeling/models/training_timestamp/`

- Train.py accepts a training/evaluation configuration name.  These configurations are located in `behavior_modeling/config`.

## Inference (`inference.py`)

Inference runs on unlabeled data and serializes results.



