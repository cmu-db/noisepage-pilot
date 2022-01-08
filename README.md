# NoisePage Pilot

This repository contains the pilot components for the [NoisePage DBMS](https://noise.page/).

## Quickstart

1. Install necessary packages.
    - `pip3 -r requirements.txt`
2. List all the tasks.
    - `doit list`
3. Select and run a doit task from the task list, e.g. `doit action_recommendation`.  Task dependencies are executed automatically.

## Sample Doit Tasks

The following is a list of some relevant doit tasks and their parameters:

- Action Recommendation
  - The following task will begin picking indexes: `doit action_recommendation`.
- Behavior Modeling
  - `doit behavior --datagen` runs behavior model training data generation using TScout, Benchbase and Postgres.  Requires `sudo` permissions for TScout.
  - `doit behavior --diff` performs training data differencing (subtracting child-plan costs).
  - `doit behavior --train` trains, evaluates, and serializes models along with their evaluations and predictions.
  - `doit behavior --all` is equivalent to `doit behavior --datagen --diff --train`.
  - Any combination of the above flags can be used.
  - Configure data generation, model training, Benchbase, and Postgres in `noisepage-pilot/config/behavior`.
  - Training data differencing and model training default to using the most recent experiment data.
  - Additional behavior modeling documentation is available at `noisepage-pilot/behavior/README.md`.

## Background

- Self-Driving DBMS = Workload Forecasting + Behavior Modeling + Action Planning.
  - Workload Forecasting: `forecast` folder.
  - Modeling: WIP.
  - Action Planning: `action` folder.
  - See [^electricsheep], [^15799] for more details.

## References

[^electricsheep]: Make Your Database System Dream of Electric Sheep: Towards Self-Driving Operation.

    ```
    @article{pavlo21,
    author = {Pavlo, Andrew and Butrovich, Matthew and Ma, Lin and Lim, Wan Shen and Menon, Prashanth and Van Aken, Dana and Zhang, William},
    title = {Make Your Database System Dream of Electric Sheep: Towards Self-Driving Operation},
    journal = {Proc. {VLDB} Endow.},
    volume = {14},
    number = {12},
    pages = {3211--3221},
    year = {2021},
    url = {https://db.cs.cmu.edu/papers/2021/p3211-pavlo.pdf},
    }
    ```

[^15799]: https://15799.courses.cs.cmu.edu/spring2022/
