# Quickstart

1. Clone this NoisePage-Pilot repo.
2. Install PostgreSQL 14.
3. (Optional but recommended) Tune your PostgreSQL instance with something like `PGTune`.
4. As the PostgreSQL user,
   1. `sudo su postgres`
   2. `psql`
   3. `create user project1user with superuser encrypted password 'project1pass';`
5. Create a new GitHub repo that contains one file, `dodo.py`, which contains the following:
```python
def task_project1():
    return {
        # A list of actions. This can be bash or Python callables.
        "actions": [
            'echo "Faking action generation."',
            'echo "SELECT 1;" > actions.sql',
            'echo "SELECT 2;" >> actions.sql',
            'echo \'{"VACUUM": true}\' > config.json',
        ],
        # Always rerun this task.
        "uptodate": [False],
    }
```
5. Add your GitHub repo to `./project/project1.sh` under the `STUDENTS` section, format `git_url,andrew_id`.
6. Run `./project/project1.sh`.

This sets up the same infrastructure used for grading.  
At a high-level, this is how grading works:

```python
def grade():
    while True: # Up to some timeout T1.
        exitcode = grade_iteration()
        if exitcode != 0:
            break

def grade_iteration():
    # Dump current DB.
    # STUDENT CODE: run `doit project1` to generate actions.csv, config.json; up to another timeout T2.
    # Restore DB from dump.
    # Apply actions from actions.csv.
    # VACUUM if requested from config.json.
    # Run BenchBase to evaluate the current configuration.
```

## Stuff to know

- All `doit` commands assume they are executed from the project root.
- You only need to know about these commands:
    - `doit benchbase_clone`: clone BenchBase.
    - `doit benchbase_run`: run BenchBase.
    - `doit project1_disable_logging`: disable logging and restart PostgreSQL.
    - `doit project1_enable_logging`: enable logging and restart PostgreSQL.
    - Try running `doit help benchbase_clone`, for example, to see the arguments it supports.
- We basically use `doit` as a powerful domain-specific language that we invoke from Bash scripts.

For example, this is how you can collect a workload trace for ePinions with a small scalefactor of 1:

```bash
# Clone Andy's BenchBase.
doit benchbase_clone --repo_url="https://github.com/apavlo/benchbase.git" --branch_name="main"
cp ./build/benchbase/config/postgres/15799_starter_config.xml ./config/behavior/benchbase/epinions_config.xml

# Generate the ePinions config file.
mkdir -p artifacts/project/
cp ./config/behavior/benchbase/epinions_config.xml ./artifacts/project/epinions_config.xml
xmlstarlet edit --inplace --update '/parameters/url' --value "jdbc:postgresql://localhost:5432/project1db?preferQueryMode=extended" ./artifacts/project/epinions_config.xml
xmlstarlet edit --inplace --update '/parameters/username' --value "project1user" ./artifacts/project/epinions_config.xml
xmlstarlet edit --inplace --update '/parameters/password' --value "project1pass" ./artifacts/project/epinions_config.xml
xmlstarlet edit --inplace --update '/parameters/scalefactor' --value "1" ./artifacts/project/epinions_config.xml

# Load ePinions.
doit benchbase_run --benchmark="epinions" --config="./artifacts/project/epinions_config.xml" --args="--create=true --load=true"

# Collect the workload trace of executing ePinions.
doit project1_enable_logging
doit benchbase_run --benchmark="epinions" --config="./artifacts/project/epinions_config.xml" --args="--execute=true"
doit project1_disable_logging

sudo ls -lah /var/lib/postgresql/14/main/log/ | grep csv
```