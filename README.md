# NoisePage Pilot

This repository contains the pilot components for the [NoisePage DBMS](https://noise.page/).

## Quickstart

1. Install necessary packages.
    - `pip3 -r requirements.txt`
2. List all the tasks.
    - `doit list`
3. Task dependencies are executed automatically.  
   The following command will start picking indexes.
    - `doit action_recommendation`
4. Run behavior modeling using: 
    - `doit behavior --datagen --diff --train`
    - This will generate TScout data, perform plan differencing, and train/evaluate/save models.
    - Any of these arguments can be removed.
    - Behavior modeling currently defaults to the latest experimental data for plan differencing and model training.
    - You can configure data generation, model training, Benchbase, and Postgres in `noisepage-pilot/config/behavior`

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
