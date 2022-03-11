set -ex

# Initialize all submodules.
git submodule update --init --recursive

# Install all dependencies in case requirements.txt has changed since last run.
# TODO(wz2): We may want to lock the versions.
pip3 install --upgrade -r requirements.txt

# Various steps may require sudo.
sudo --validate

# Increase file descriptor limit for TScout.
ulimit -n 8096

# Clean up the result of previous runs, if any.
doit ci_clean_slate
sudo pkill postgres || true
doit behavior_microservice_kill

# Behavior models need to be trained.
doit behavior_generate_workloads
doit behavior_execute_workloads
doit behavior_perform_plan_diff

# By default, don't use featurewiz since that will increase the training time.
# doit behavior_train --use_featurewiz=True
doit behavior_train

# Forecast: generate training data.
# This is annoying to automate because we need to enable/disable
# logging and also customize the workload we want, so we don't.
doit noisepage_init
doit benchbase_overwrite_config
doit benchbase_bootstrap_dbms
doit benchbase_prewarm_install
doit benchbase_run --benchmark="tpcc" --args="--create=true --load=true"
doit noisepage_enable_logging
doit benchbase_run --benchmark="tpcc" --args="--execute=true"
doit noisepage_disable_logging

# Because BenchBase scales down the workload, we would end up predicting nothing by default.
# We hack around it by deleting the last few query log files.
doit noisepage_truncate_log

# Predict the forecast manually because we need to control the time for now.
# Otherwise it would default to predicting [now, now+1 minute].
doit forecast_predict

##############################################################################
# After this line, both forecast and behavior models should be ready.
##############################################################################

# Start the inference microservice.
doit behavior_microservice

# Start with a fresh copy of NoisePage.
doit noisepage_init

# Install Hutch.
doit noisepage_hutch_install

# Install HypoPG.
doit action_selection_hypopg_install
# Bootstrap the DBMS for action recommendation.
doit action_recommendation_bootstrap_dbms

# We need to load the same schema as the forecast.
# TODO(WAN): Editing XML can be such a pain...
mkdir -p artifacts/demo/
sudo apt-get install xmlstarlet
cp ./config/behavior/benchbase/tpcc_config.xml ./artifacts/demo/mvp_tpcc_config.xml
xmlstarlet edit --inplace --update '/parameters/url' --value 'jdbc:postgresql://localhost:5432/np_as_spiel?preferQueryMode=extended' ./artifacts/demo/mvp_tpcc_config.xml
xmlstarlet edit --inplace --update '/parameters/username' --value 'np_as_spiel_user' ./artifacts/demo/mvp_tpcc_config.xml
xmlstarlet edit --inplace --update '/parameters/password' --value 'np_as_spiel_pass' ./artifacts/demo/mvp_tpcc_config.xml
doit benchbase_run --benchmark="tpcc" --config="./artifacts/demo/mvp_tpcc_config.xml" --args="--create=true --load=true"
# And then we need to drop those two TPC-C indexes.
PGPASSWORD=np_as_spiel_pass ./artifacts/noisepage/psql -h localhost -d np_as_spiel -U np_as_spiel_user -c "DROP INDEX idx_customer_name;"
PGPASSWORD=np_as_spiel_pass ./artifacts/noisepage/psql -h localhost -d np_as_spiel -U np_as_spiel_user -c "DROP INDEX idx_order;"

# Manually generate actions because we want to pass the --filter-tables flag.
# Filtering because infer() as a microservice is, unsurprisingly, extremely slow.
doit action_generation --args="--min-num-cols 1 --max-num-cols 4 --filter-tables"

# Start picking indexes.
doit action_recommendation --database_game_args="--use_microservice"

sudo --reset-timestamp
