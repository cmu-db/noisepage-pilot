#!/bin/bash
set -ex

###
#
# A given workload folder contains the following pieces of information.
#
# 1. A `config.yaml` file that is structured as follows:
#       benchmark: [name of the benchmark to execute]
#       pg_prewarm: True/False [whether to pg_prewarm prior to each benchbase run]
#       pg_analyze: True/False [whether to pg_analyze prior to each benchbase run]
#       pg_configs: [List of files to posgresql.conf files to switch]
#       benchbase_configs: [List of benchbase XML configs for each run]
#
# 2. Relevant postgresql.conf files that should be used for execution
# 3. Relevant BenchBase configuration XMLs that should be used for execution
#
# KNOWN CAVEAT:
#
# 1. `run_workloads.sh` does not handle scale factor shifting between benchbase_configs.
# The benchmark database is loaded with the first benchbase config and then reused
# afterwards. As such, this script does not respect the scale factor of later
# BenchBase configurations (workload distribution & other parameters are respected).
#
# 2. `run_workloads.sh` does not currently support loading multiple databases and having
# benchbase_configs execute differente benchmarks. However, support for this is not difficult
# to add if needed.
#
###

# Various steps may require sudo.
sudo --validate

# Check that the file descriptor limit is at least 8096.
limit=$(ulimit -n)
if [ "$limit" -lt 8096 ];
then
    echo "TScout requires [ulimit -n] to return at least 8096."
    exit 1
fi

help() {
    echo "run_workloads.sh [ARGUMENTS]"
    echo ""
    echo "Arguments"
    echo "---------"
    echo "--workloads=[Directory containing all workloads to execute]"
    echo "--output-dir=[Directory to write workload outputs]"
    echo "--pgdata=[Directory of postgres instance to use]"
    echo "--benchbase=[Directory of benchbase installation]"
    echo "--pg_binaries=[Directory containing postgres binaries]"
    exit 1
}

# Parsing --[arg_name]=[arg_value] constructs
arg_parse() {
    for i in "$@"; do
        case $i in
            -workloads=*|--workloads=*)
                WORKLOADS_DIRECTORY="${i#*=}"
                shift # past argument=value
                ;;
            -output-dir=*|--output-dir=*)
                OUTPUT_DIRECTORY="${i#*=}"
                shift # past argument=value
                ;;
            -pgdata=*|--pgdata=*)
                PGDATA_LOCATION="${i#*=}"
                shift # past argument with no value
                ;;
            -benchbase=*|--benchbase=*)
                BENCHBASE_LOCATION="${i#*=}"
                shift # past argument with no value
                ;;
            -pg_binaries=*|--pg_binaries=*)
                PG_BINARIES_LOCATION="${i#*=}"
                PG_CTL_LOCATION="${PG_BINARIES_LOCATION}/pg_ctl"
                PSQL_LOCATION="${PG_BINARIES_LOCATION}/psql"
                shift # past argument with no value
                ;;
            -*)
                echo "Unknown option $i"
                help
                ;;
            *)
                ;;
        esac
    done
}

arg_validate() {
    if [ -z ${WORKLOADS_DIRECTORY+x} ] ||
       [ -z ${OUTPUT_DIRECTORY+x} ] ||
       [ -z ${PGDATA_LOCATION+x} ] ||
       [ -z ${BENCHBASE_LOCATION+x} ];
    then
        help
        exit 1
    fi

    if [ ! -d "${WORKLOADS_DIRECTORY}" ];
    then
        echo "Specified workload directory ${WORKLOADS_DIRECTORY} does not exist."
    fi

    if [ ! -d "${BENCHBASE_LOCATION}" ];
    then
        echo "Specified benchbase ${BENCHBASE_LOCATION} does not exist."
    fi

    if [ ! -f "${PG_CTL_LOCATION}" ];
    then
        echo "Specified pg_ctl ${PG_CTL_LOCATION} does not exist."
    fi

    if [ ! -f "${PSQL_LOCATION}" ];
    then
        echo "Specified pg_ctl ${PSQL_LOCATION} does not exist."
    fi
}


# Parse all the input arguments to the bash script.
arg_parse "$@"
# Validate all the input arguments passed to the bash script.
arg_validate
# Record the current timestamp
ts=$(date '+%Y-%m-%d_%H-%M-%S')
echo "Starting workload execution ${ts}"

# Get the absolute file path to the pg_ctl executable
pg_ctl=$(realpath "${PG_CTL_LOCATION}")
psql=$(realpath "${PSQL_LOCATION}")

# Kill any running postgres and/or TScout instances.
pkill -i postgres || true
pkill -i tscout || true

modes=("train" "eval")
for mode in "${modes[@]}"; do
    output_folder="${OUTPUT_DIRECTORY}/experiment-${ts}/${mode}"
    workload_directory="${WORKLOADS_DIRECTORY}/${mode}"
    for workload in "${workload_directory}"/*; do
        echo "Executing ${workload} for ${mode}"

        # Create the output directory for this particular benchmark invocation.
        benchmark_suffix=$(basename "${workload}")
        benchmark_output="${output_folder}/${benchmark_suffix}"
        mkdir -p "${benchmark_output}"

        # Parse the config.yaml file that describes the experiment.
        # The description for the config.yaml and the keys populated are described
        # in behavior/datagen/generate_workloads.py.
        config_yaml=$(realpath "${workload}"/config.yaml)
        eval "$(niet -f eval . "${config_yaml}")"

        # shellcheck disable=2154 # populated by niet
        if [ ${#benchbase_configs[@]} != ${#pg_configs[@]} ];
        then
            echo "Found configuration file ${config_yaml} where configurations are not aligned."
            exit 1
        fi

        if [ ${#benchbase_configs[@]} == 0 ];
        then
            echo "Found configuration file {$config_yaml} containing empty experiment."
            exit 1
        fi

        for i in "${!benchbase_configs[@]}";
        do
            postgresql_path=$(realpath "${pg_configs[$i]}")
            benchbase_config_path=$(realpath "${benchbase_configs[$i]}")

            if [ "$i" -eq 0 ];
            then
                # If we're executing a new experiment, then we want to completely
                # the database instance. This is done by invoking `noisepage_init`.
                doit noisepage_init --config="${postgresql_path}"
                doit benchbase_bootstrap_dbms

                # Create the database and load the database
                # shellcheck disable=2154 # populated by niet
                doit benchbase_run --benchmark="${benchmark}" --config="${benchbase_config_path}" --args="--create=true --load=true --execute=false"

                # Remove existing logfiles, if any exist.
                ${pg_ctl} stop -D "${PGDATA_LOCATION}" -m smart
                rm -rf "${PGDATA_LOCATION}/log/*.csv"
                rm -rf "${PGDATA_LOCATION}/log/*.log"
                rm -rf "${BENCHMARK_LOCATION}/results/*"

                # Then restart the instance.
                ${pg_ctl} start -D "${PGDATA_LOCATION}"
            else
                doit noisepage_swap_config --config="${postgresql_path}"

                # We don't need to bootstrap the benchbase database here because
                # recovery should reload the entire benchbase database state.
            fi

            # shellcheck disable=2154 # populated by niet
            if [ "$pg_prewarm" != 'False' ];
            then
                # If pg_prewarm is specified, then invoke pg_prewarm on the benchmark's tables.
                doit benchbase_prewarm_install
                doit behavior_pg_prewarm_benchmark --benchmark="${benchmark}"
            fi

            # shellcheck disable=2154 # populated by niet
            if [ "$pg_analyze" != 'False' ];
            then
                # If pg_analyze is specified, then run ANALYZE on the benchmark's tables.
                doit behavior_pg_analyze_benchmark --benchmark="${benchmark}"
            fi

            # Initialize TScout. We currently don't have a means by which to check whether
            # TScout has successfully attached to the instance. As such, we (wait) 5 seconds.
            append=$( [[ $i != "0" ]] && echo "True" || echo "False" )
            doit tscout_init --output_dir="${benchmark_output}" --wait_time=5 --append="${append}"

            # Execute the benchmark
            doit benchbase_run --benchmark="${benchmark}" --config="${benchbase_config_path}" --args="--execute=true"

            # Shutdown TScout and take ownership of the results.
            doit tscout_shutdown --output_dir="${benchmark_output}" --wait_time=10 --flush_time=5

            # Since pg_stats can change in between benchmark invocations, pg_stats is written out
            # to a file after each benchmark invocation within an experiment. The CSV file is
            # suffixed by the benchmark index.
            stats_file="${benchmark_output}/pg_stats.csv.${i}"
            ${psql} --dbname=benchbase --csv --command="SELECT * FROM pg_stats;" > "${stats_file}"

            # Similarly, we move the postgres log file to the experiment output directory if it
            # exists. The log file is also suffixed by this benchmark index.
            log=${PGDATA_LOCATION}/log
            ${pg_ctl} stop -D "${PGDATA_LOCATION}" -m smart
            if [ -d "${log}" ];
            then
                mv "${PGDATA_LOCATION}/log" "${benchmark_output}/log.${i}"
            fi

            # Similarly, we move the corresponding benchmark's execution log from BenchBase to the
            # experiment output directory, with the results folder suffixed by the benchmark index.
            mv "${BENCHBASE_LOCATION}/results" "${benchmark_output}/results.${i}"
        done

        echo "Executed ${workload} for ${mode}"
    done
done

ts=$(date '+%Y-%m-%d_%H-%M-%S')
echo "Finished workload execution ${ts}"
