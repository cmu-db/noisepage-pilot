# Behavior Modeling

This document details the core components of behavior modeling and how to use them.

## Pipeline

This diagram details the general workflow.

![Behavior Modeling Diagram](../docs/behavior/behavior_modeling_pipeline.svg)

## Postgres Plan Nodes

The following is a list of Postgres query plan nodes, each of which are profiled by TScout.

- Agg
- Append
- CteScan
- CustomScan
- ForeignScan
- FunctionScan
- Gather
- GatherMerge
- Group
- HashJoinImpl
- IncrementalSort
- IndexOnlyScan
- IndexScan
- Limit
- LockRows
- Material
- MergeAppend
- MergeJoin
- ModifyTable
- NamedTuplestoreScan
- NestLoop
- ProjectSet
- RecursiveUnion
- Result
- SampleScan
- SeqScan
- SetOp
- Sort
- SubPlan
- SubqueryScan
- TableFuncScan
- TidScan
- Unique
- ValuesScan
- WindowAgg
- WorkTableScan

## BenchBase Benchmark Databases

The following BenchBase benchmarks have been tested to work with behavior modeling.

- AuctionMark
- Epinions
- SEATS
- SIBench
- SmallBank
- TATP
- TPC-C
- Twitter
- Voter
- Wikipedia
- YCSB

Caveats:

- Various benchmarks yield slightly different OUs for a BenchBase run with the same configuration. Cause unknown.
- If the BenchBase experiment duration is too short, you may not get data for all OUs.
- Using pg_stat_statements and auto_explain will affect benchmark statistics and model performance.
    - These are only intended for debugging.
- TPC-H support is blocked on the [native loader](https://github.com/cmu-db/benchbase/pull/99) being merged.
- Epinions is missing results for the Materialize OU in the plan generated for `GetReviewsByUser`.
    - `SELECT * FROM review r, useracct u WHERE u.u_id = r.u_id AND r.u_id=$1 ORDER BY rating LIMIT 10`

## Resource Consumption Metrics

The following is a list of resource consumption metrics that TScout collects and the operating unit models predict.

- cpu_cycles
- instructions
- cache_references
- cache_misses
- ref_cpu_cycles
- network_bytes_read
- network_bytes_written
- disk_bytes_read
- disk_bytes_written
- memory_bytes
- elapsed_us

## Operating Unit (OU) Model Variants

- Tree-based
    - dt
    - rf - good performance
    - gbm - good performance
- Multi-layer perceptron
    - mlp
- Generalized linear models
    - lr
    - huber
    - mt_lasso
    - lasso
    - mt_elastic
    - elastic

## Training Data

- The data generator automatically creates both training and evaluation sets.  This allows for easy evaluation on non-training data.
- The alternative is to have the data stored jointly and maintain indexes or nested directories partitioning training and evaluation data.  This is typically only worth doing if training data is scarce and expensive because it allows for more flexible experimentation with minimal data.  This isn't a concern here, so we fully separate train and evaluation datasets.  This also avoids issues of intra-run data leakage; i.e. the data within a given round not being I.I.D.
- In the future, it may be best to store this data in a database because it is tabular and SQL-based access will allow for more flexible and dynamic construction of training/evaluation sets.  This was not a concern in MB2 because the data was statically created one time by handwritten microbenchmarks and all columns for all OUs were identical.

## Training (`train.py`)

Trains, evaluates, and serializes models, saving all results to `noisepage-pilot/data/behavior/models/${training_timestamp}/`

- Train.py accepts a training/evaluation configuration name.  These configurations are located in `noisepage-pilot/config/behavior/`.

## Inference (`inference.py`) (WIP)

Inference runs on unlabeled data and serializes results.
Wan and Garrison are working on the API for this.

## References

See [^mb2] for more details.

[^mb2]: MB2: Decomposed Behavior Modeling for Self-Driving Database Management Systems

    ```
    @article{ma21,
    author = {Ma, Lin and Zhang, William and Jiao, Jie and Wang, Wuwen and Butrovich, Matthew and Lim, Wan Shen and Menon, Prashanth and Pavlo, Andrew},
    title = {MB2: Decomposed Behavior Modeling for Self-Driving Database Management Systems},
    journal = {SIGMOD},
    year = {2021},
    url = {https://www.cs.cmu.edu/~malin199/publications/2021.mb2.sigmod.pdf},
    }
    ```
