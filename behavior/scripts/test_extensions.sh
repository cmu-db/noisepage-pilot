#!/bin/bash


rm -rf data
mkdir data
./build/bin/pg_ctl initdb -D data
cp ./cmudb/behavior_modeling/config/datagen/postgres/postgresql.conf ./data
./build/bin/pg_ctl -D data start
./build/bin/createdb 'test'
# ./build/bin/psql -d 'test' -c "CREATE EXTENSION pg_stat_statements;"
# "ALTER SYSTEM SET configuration_parameter { TO | = } { value | 'value' | DEFAULT };"
# ./build/bin/psql -d 'test' -c "EXPLAIN (VERBOSE) SELECT 1;"
# ./build/bin/psql -d 'test' -c "SELECT * FROM pg_stat_statements;"
./build/bin/pg_ctl -D data stop


