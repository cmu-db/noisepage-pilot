# Forecast

## Background

- Based on QueryBot5000 [^querybot5000].
- Conceptual steps:
    1. Preprocess query log files into (query templates, input parameters).
    2. Cluster query templates by arrival history.
    3. Forecast the arrival rate of the query templates. (TODO(WAN): Mike, add description?)
    4. Reverse-map the forecasted cluster back into SQL queries by sampling the input parameters.

## Folder contents

- preprocessor.py
    - Input: PostgreSQL query log files.
    - Output: pandas dataframes.
    - Purpose: Processes PostgreSQL query log files into pandas dataframes.
- clusterer.py
    - Input: preprocessor output.
    - Output: forecasted SQL queries.
    - Performs clustering (TODO(WAN): Pending Mike).

## References

[^querybot5000]: Query-based Workload Forecasting for Self-Driving Database Management Systems.

    ```
    @inproceedings{ma18,
    author = {Ma, Lin and Van Aken, Dana and Hefny, Ahmed and Mezerhane, Gustavo and Pavlo, Andrew and Gordon, Geoffrey J.},
    title = {Query-based Workload Forecasting for Self-Driving Database Management Systems},
    booktitle = {Proceedings of the 2018 International Conference on Management of Data},
    series = {SIGMOD '18},
    year = {2018},
    pages = {631--645},
    numpages = {15},
    doi = {10.1145/3183713.3196908},
    url = {https://db.cs.cmu.edu/papers/2018/mod435-maA.pdf},
    }
    ```
