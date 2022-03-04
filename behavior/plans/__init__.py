from __future__ import annotations

# These OUs are currently blacklisted from being handled by plan diff-ing. Note that Gather/GatherMerge
# are only present for data where there is a parallel query operation. If in the future, we want to
# support differencing on parallel data, we would require the following:
#
# 1- TScout instrumentation such that the PID for all OUs in a given query are the same regardless if it
#    is being executed by the Leader or the parallel workers.
#
# 2- Define a mechanism by which metrics should be altered.
BLACKLIST_OUS = ["Gather", "GatherMerge"]

# OUs read from CSV have distinct schemas since each OU contains different features that are extracted.
# The `COMMON_SCHEMA` defined below standardizes a set of features for each OU in order to perform
# differencing correctly.
COMMON_SCHEMA: list[str] = [
    # These identify the (OU, row) of this particular datapoint in the unified frame.
    # ou_index corresponds to an index into PLAN_NODE_NAMES.
    "ou_index",
    "data_id",
    # These identify a unique query invocation.
    # `query_id` is a post-parse representation of the user's query.
    #
    # `pid` is used to identify a given terminal/client session.
    #
    # `statement_timestamp` identifies a given invocation on a given terminal. This is because postgres
    # invokes SetCurrentStatementStartTimestamp() on every simple query invocation and also on every
    # EXECUTE in the extended query protocol.
    "query_id",
    "statement_timestamp",
    "pid",
    # Start Time and End Time of the given OU invocation.
    "start_time",
    "end_time",
    # These are used to reconstruct the plan tree.
    "plan_node_id",
    "left_child_plan_node_id",
    "right_child_plan_node_id",
]

# The set of columns to standardize the input data's columns on. A column that ends with any
# of the values defined below is renamed to the defined value.
STANDARDIZE_COLUMNS: list[str] = [
    "query_id",
    "plan_node_id",
    "left_child_plan_node_id",
    "right_child_plan_node_id",
    "startup_cost",
    "total_cost",
    "start_time",
    "end_time",
    "statement_timestamp",
    "pid",
]

# Given a 2D DataFrame following the COMMON_SCHEMA, these give the column offsets of the plan node,
# the left child plan node, the right child plan node, and where the target columns begin.
PLAN_NODE_ID_SCHEMA_INDEX = COMMON_SCHEMA.index("plan_node_id")
LEFT_CHILD_PLAN_NODE_ID_SCHEMA_INDEX = COMMON_SCHEMA.index("left_child_plan_node_id")
RIGHT_CHILD_PLAN_NODE_ID_SCHEMA_INDEX = COMMON_SCHEMA.index("right_child_plan_node_id")
TARGET_START_SCHEMA_INDEX = len(COMMON_SCHEMA)
