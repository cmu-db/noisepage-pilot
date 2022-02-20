#!/usr/bin/env bash
# This script should be run from the root noisepage-pilot folder.

# TODO(Matt): Update student list.
# Define all the students, format is "git_url,andrew_id".
STUDENTS=(
  'git@github.com:lmwnshn/S22-15799.git,wanshenl'
)

# TODO(Matt): Update benchmark list with workloads.
# Define all the benchmarks, format is "benchmark,workload_csv".
BENCHMARKS=(
  'epinions,/tmp/epinions.csv'
  #'indexjungle,/tmp/indexjungle.csv'
)

# Set VERBOSITY to 0 for grading, 2 for development.
VERBOSITY=2

# You should set up a user like this. The script will handle creating the database.
# postgres=# create user project1user with superuser encrypted password 'project1pass';
# CREATE ROLE
# Additionally, export these variables so that they are available to the grading subshells.
export DB_USER="project1user"
export DB_PASS="project1pass"
export DB_NAME="project1db"

# TODO(Matt): Finalize these timeouts.
# TA: Total time that the grading script can run per (student, benchmark).
export TIME_GRADING_TOTAL="3m"
# Student: Maximum time allowed per (student, benchmark, action generation).
export TIME_ACTION_GENERATION="10m"

# Setup the database using the global constants.
_setup_database() {
  # Drop the project database if it exists.
  PGPASSWORD=${DB_PASS} dropdb --host=localhost --username=${DB_USER} --if-exists ${DB_NAME}
  # Create the project database.
  PGPASSWORD=${DB_PASS} createdb --host=localhost --username=${DB_USER} ${DB_NAME}
}

_setup_benchmark() {
  benchmark="${1}"

  echo "Loading: ${benchmark}"

  # Modify the BenchBase benchmark configuration.
  mkdir -p artifacts/project/
  cp ./config/behavior/benchbase/${benchmark}_config.xml ./artifacts/project/${benchmark}_config.xml
  xmlstarlet edit --inplace --update '/parameters/url' --value "jdbc:postgresql://localhost:5432/${DB_NAME}?preferQueryMode=simple" ./artifacts/project/${benchmark}_config.xml
  xmlstarlet edit --inplace --update '/parameters/username' --value "${DB_USER}" ./artifacts/project/${benchmark}_config.xml
  xmlstarlet edit --inplace --update '/parameters/password' --value "${DB_PASS}" ./artifacts/project/${benchmark}_config.xml
  xmlstarlet edit --inplace --update '/parameters/scalefactor' --value "1" ./artifacts/project/${benchmark}_config.xml
  xmlstarlet edit --inplace --update '/parameters/works/work/time' --value "30" ./artifacts/project/${benchmark}_config.xml
  xmlstarlet edit --inplace --update '/parameters/works/work/rate' --value "unlimited" ./artifacts/project/${benchmark}_config.xml

  # Load the benchmark into the project database.
  doit --verbosity ${VERBOSITY} benchbase_run --benchmark="${benchmark}" --config="./artifacts/project/${benchmark}_config.xml" --args="--create=true --load=true"
}

_dump_database() {
  dump_path="${1}"

  # Dump the project database into directory format.
  rm -rf "./${dump_path}"
  PGPASSWORD=$DB_PASS pg_dump --host=localhost --username=$DB_USER --format=directory --file=./${dump_path} $DB_NAME

  echo "Dumped database to: ${dump_path}"
}

_restore_database() {
  dump_path="${1}"

  # Restore the project database from directory format.
  PGPASSWORD=${DB_PASS} pg_restore --host=localhost --username=$DB_USER --clean --if-exists --dbname=${DB_NAME} ./${dump_path}

  echo "Restored database from: ${dump_path}"
}

_clear_log_folder() {
  sudo bash -c "rm -rf /var/lib/postgresql/14/main/log/*"
  echo "Cleared all query logs."
}

_copy_logs() {
  save_path="${1}"

  # TODO(WAN): Is there a way to ensure all flushed?
  sleep 10
  sudo bash -c "cat /var/lib/postgresql/14/main/log/*.csv > ${save_path}"
  echo "Copied all query logs to: ${save_path}"
}

_grade_iteration() {
  submission_path="${1}"
  benchmark="${2}"
  workload_csv="${3}"
  iteration="${4}"

  actions_file="./actions.sql"
  config_file="./config.json"
  results_folder="${submission_path}/${benchmark}/iteration_${iteration}"
  dump_folder="./${benchmark}_dump_tmp"

  echo "Grading benchmark ${benchmark} workload ${workload_csv} iteration ${iteration}: ${submission_path}"
  set -e

  # cd to the student folder.
  cd ${submission_path} || exit 1
  # Dump the current database state.
  _dump_database "${dump_folder}"
  # Run action generation with a timeout.
  timeout ${TIME_ACTION_GENERATION} doit project1 --workload_csv="${workload_csv}" --timeout="${TIME_ACTION_GENERATION}"
  # Restore the database state.
  _restore_database "${dump_folder}"
  # Remove the temporary dump folder.
  rm -rf "./${dump_folder}"
  # Apply the generated actions.
  PGPASSWORD=${DB_PASS} psql --host=localhost --username=${DB_USER} --dbname=${DB_NAME} --file="./${actions_file}"
  # Run VACUUM FULL if requested.
  if [ "$(jq -r '.VACUUM' ${config_file})" == "true" ]; then
    PGPASSWORD=${DB_PASS} psql --host=localhost --username=${DB_USER} --dbname=${DB_NAME} --command="VACUUM FULL;"
  fi
  # Reset the DBMS's internal metrics.
  PGPASSWORD=${DB_PASS} psql --host=localhost --username=${DB_USER} --dbname=${DB_NAME} --command="SELECT pg_stat_reset();"

  # cd to the original root folder again.
  cd - 1>/dev/null || exit 1
  # Evaluate the performance on the workload.
  _clear_log_folder
  doit project1_enable_logging
  doit --verbosity ${VERBOSITY} benchbase_run --benchmark="${benchmark}" --config="./artifacts/project/${benchmark}_config.xml" --args="--execute=true"
  doit project1_disable_logging

  # Yoink the result files.
  mkdir -p "${results_folder}"
  mv ./artifacts/benchbase/results/* "${results_folder}"
  # Yoink the student stuff too because why not.
  mv "./${submission_path}/${actions_file}" "${results_folder}"
  mv "./${submission_path}/${config_file}" "${results_folder}"
  # Save the workload trace for the next iteration.
  _copy_logs "${results_folder}/workload.csv"

  # IF YOU WANT TO SAVE DISK SPACE, YOU SHOULD UNCOMMENT THE FOLLOWING LINE.
  # rm ${workload_csv}

  # Return cleanly.
  return 0
}

_grade() {
  submission_path="${1}"
  benchmark="${2}"
  workload_csv="${3}"

  # Make a private copy of workload CSV.
  bootstrap_folder="${submission_path}/${benchmark}/bootstrap"
  mkdir -p ${bootstrap_folder}
  cp ${workload_csv} ${bootstrap_folder}/workload.csv
  workload_csv="$(pwd)/${bootstrap_folder}/workload.csv"

  # Unfortunately, timeout doesn't work on special Bash constructs.
  # We export the function and wrap everything in a subshell.
  export -f _grade_iteration _dump_database _restore_database
  (
    iteration="1"
    keep_going="true"

    stop_grading() {
      keep_going="false"
    }
    # 124 is the code returned by `timeout` on timeout.
    trap stop_grading SIGINT

    while [ "${keep_going}" == "true" ]; do
      _grade_iteration "${submission_path}" "${benchmark}" "${workload_csv}" "${iteration}"
      case $? in
      0)
        # Program exited normally, looping for another iteration.
        # Therefore we do not break.
        ;;
      *)
        # Program exited abnormally, terminate.
        echo "Premature exit, exit code: $?"
        break
        ;;
      esac
      workload_csv="$(pwd)/${submission_path}/${benchmark}/iteration_${iteration}/workload.csv"
      iteration=$((iteration + 1))
    done
  ) &
  grader_pid=$!
  (
    echo "Will sleep for '${TIME_GRADING_TOTAL}' before termination."
    sleep ${TIME_GRADING_TOTAL} &
    wait
    echo "Time's up."
    kill -s INT ${grader_pid}
  ) &
  sleep_pid=$!
  wait ${grader_pid}
  kill ${sleep_pid} 2>/dev/null
}

kill_descendant_processes() {
  local pid="$1"
  local and_self="${2:-false}"
  if children="$(pgrep -P "$pid")"; then
    for child in $children; do
      kill_descendant_processes "$child" true
    done
  fi
  if [[ "$and_self" == true ]]; then
    sudo kill -9 "$pid"
  fi
}

exit_cleanly() {
  kill_descendant_processes $$
}

main() {
  trap exit_cleanly SIGINT
  trap exit_cleanly SIGTERM

  set -e
  # Ask for sudo now, we're going to need it.
  sudo --validate
  # jq to parse parameters from student scripts.
  sudo apt-get -qq install jq
  # xmlstarlet to edit BenchBase XML configurations.
  sudo apt-get -qq install xmlstarlet

  # Clean up before running.
  rm -rf ./artifacts/
  rm -rf ./build/

  # Use Andy's version of BenchBase.
  doit benchbase_clone --repo_url="https://github.com/apavlo/benchbase.git" --branch_name="main"
  cp ./build/benchbase/config/postgres/15799_starter_config.xml ./config/behavior/benchbase/epinions_config.xml
  cp ./build/benchbase/config/postgres/15799_indexjungle_config.xml ./config/behavior/benchbase/indexjungle_config.xml

  benchmark_dump_folder="./artifacts/project/dumps"
  # Create the folder for all the benchmark dumps.
  mkdir -p "./${benchmark_dump_folder}"

  for benchmark_spec in "${BENCHMARKS[@]}"; do
    while IFS=',' read -r benchmark workload_csv; do
      benchmark_dump_path="./${benchmark_dump_folder}/${benchmark}_primary"

      # Create the project database.
      _setup_database
      # Load the benchmark data.
      _setup_benchmark "${benchmark}"
      # Dump the project database to benchmark_primary.
      _dump_database "${benchmark_dump_path}"
      # Generate the base workload CSV.
      _clear_log_folder
      doit project1_enable_logging
      doit benchbase_run --benchmark="${benchmark}" --config="./artifacts/project/${benchmark}_config.xml" --args="--execute=true"
      doit project1_disable_logging
      _copy_logs "/tmp/${benchmark}.csv"
      _clear_log_folder
      # Restore the project database.
      _restore_database "${benchmark_dump_path}"

      # Create a folder for student submissions, if it doesn't exist yet.
      student_submission_folder="./artifacts/project/student"
      mkdir -p "./${student_submission_folder}"

      # Preliminaries done!
      # Time to test student submissions.
      for student in "${STUDENTS[@]}"; do
        while IFS=',' read -r git_url andrew_id; do
          student_submission_path="${student_submission_folder}/${andrew_id}"
          student_submission_path="${student_submission_folder}/${andrew_id}"

          # Restore the state of the database.
          _restore_database "${benchmark_dump_path}"
          # Clone the student's submission, if it hasn't been cloned yet.
          git clone --quiet $git_url ${student_submission_path} || true
          # Grade the student submission in a subshell.
          # This avoids potential cd shenanigans.
          # Additionally, a single student grading fail shouldn't stop the harness.
          set +e
          (_grade ${student_submission_path} ${benchmark} ${workload_csv})
          set -e
          # TODO(WAN): Export grades?
        done <<<"$student"
      done
    done <<<"$benchmark_spec"
  done
}

main
exit_cleanly
