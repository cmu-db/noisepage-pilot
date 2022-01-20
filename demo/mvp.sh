set -ex

# Various steps may require sudo.
sudo --validate

# Clean up the result of previous runs, if any.
doit ci_clean_slate

# Behavior models need to be trained.
# Because the behavior code was written prior to pilot/dodo,
# datagen in particular doesn't quite fit the rest of the framework
# in that it opaquely wraps the data generation pipeline.
# This prevents it from composing with new dodo tasks.
# It's not a big deal; probably not worth refactoring right now.
doit behavior_datagen
doit behavior_train

# Forecast: generate training data.
# This is annoying to automate because we need to enable/disable
# logging and also customize the workload we want, so we don't.
doit noisepage_init
doit benchbase_overwrite_config
doit benchbase_bootstrap_dbms
doit benchbase_run --benchmark="tpcc" --args="--create=true --load=true"
doit noisepage_enable_logging
doit benchbase_run --benchmark="tpcc" --args="--execute=true"
doit noisepage_disable_logging

# Predict the forecast manually because we need to control the time for now.
# Otherwise it would default to predicting [now, now+1 minute].
# TODO(WAN): Eventually, figure out the truncation problem.
doit forecast_predict --time_start='2022-01-20 08:05:00 EST' --time_end='2022-01-20 08:30:00 EST'

##############################################################################
# After this line, both forecast and behavior models should be ready.
##############################################################################

# Start the inference microservice.
doit behavior_microservice

# Start with a fresh copy of NoisePage.
doit noisepage_init

# Install HypoPG.
doit action_selection_hypopg_install
# Bootstrap the DBMS for action recommendation.
doit action_recommendation_bootstrap_dbms

# We need to load the same schema as the forecast.
# TODO(WAN): Editing XML can be such a pain...
sudo apt-get install xmlstarlet
cp ./config/behavior/benchbase/tpcc_config.xml ./demo/mvp_tpcc_config.xml
xmlstarlet edit --inplace --update '/parameters/url' --value 'jdbc:postgresql://localhost:5432/np_as_spiel?preferQueryMode=extended' ./demo/mvp_tpcc_config.xml
xmlstarlet edit --inplace --update '/parameters/username' --value 'np_as_spiel_user' ./demo/mvp_tpcc_config.xml
xmlstarlet edit --inplace --update '/parameters/password' --value 'np_as_spiel_pass' ./demo/mvp_tpcc_config.xml
doit benchbase_run --benchmark="tpcc" --config="./demo/mvp_tpcc_config.xml" --args="--create=true --load=true"
# And then we need to drop those two TPC-C indexes.
PGPASSWORD=np_as_spiel_pass ./artifacts/noisepage/psql -h localhost -d np_as_spiel -U np_as_spiel_user -c "DROP INDEX idx_customer_name;"
PGPASSWORD=np_as_spiel_pass ./artifacts/noisepage/psql -h localhost -d np_as_spiel -U np_as_spiel_user -c "DROP INDEX idx_order;"

# Manually generate actions because we want to pass the --filter-tables flag.
# Filtering because infer() as a microservice is, unsurprisingly, extremely slow.
doit action_generation --args="--min-num-cols 1 --max-num-cols 4 --filter-tables"

# Start picking indexes.
doit action_recommendation --database_game_args="--use_microservice"

sudo --reset-timestamp
