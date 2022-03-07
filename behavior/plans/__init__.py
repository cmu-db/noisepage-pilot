from __future__ import annotations

from behavior import BASE_TARGET_COLS

# These OUs are currently blocked from being handled by plan diff-ing. Note that Gather/GatherMerge
# are only present for data where there is a parallel query operation. As such, this process will
# not output any plan data that uses any blocked OU (by extension, blocked OU data will not appear
# in the output data files).
#
# If in the future, we want to support differencing on parallel data, we would require the following:
#
# 1- TScout instrumentation such that the PID for all OUs in a given query are the same regardless if it
#    is being executed by the Leader or the parallel workers.
#
# 2- Define a mechanism by which metrics should be altered.
BLOCKED_OUS = ["Gather", "GatherMerge"]

# OUs read from CSV have distinct schemas since each OU contains different features that are extracted.
# `DIFFERENCING_SCHEMA` defines the minimal set of features common to each OU that are required to
# perform differencing correctly.
#
# If *you* ever consider adding to this list, please consider whether the feature is common to all OUs
# and whether that feature is required to perform differencing.
DIFFERENCING_SCHEMA: list[str] = (
    [
        # (ou_index, data_id) identify the (OU, row) of this particular datapoint in the unified frame.
        # ou_index corresponds to an index into PLAN_NODE_NAMES.
        "ou_index",
        "data_id",
        # (query_id, statement_timestamp, pid) identify a unique query invocation.
        # `query_id` is a post-parse representation of the user's query.
        #
        # `statement_timestamp` identifies a given invocation on a given terminal. This is because postgres
        # invokes SetCurrentStatementStartTimestamp() on every simple query invocation and also on every
        # EXECUTE in the extended query protocol.
        #
        # `pid` is used to identify a given terminal/client session.
        "query_id",
        "statement_timestamp",
        "pid",
        # Start Time and End Time of the given OU invocation.
        "start_time",
        "end_time",
        # (plan_node_id, left_child_plan_node_id, right_child_plan_node_id) are used to reconstruct the plan tree.
        "plan_node_id",
        "left_child_plan_node_id",
        "right_child_plan_node_id",
    ]
    + ["total_cost", "startup_cost"]
    + BASE_TARGET_COLS
)

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

# Given a 2D DataFrame following the DIFFERENCING_SCHEMA, these give the column offsets of the plan node,
# the left child plan node, the right child plan node, and where the target columns begin.
PLAN_NODE_ID_SCHEMA_INDEX = DIFFERENCING_SCHEMA.index("plan_node_id")
LEFT_CHILD_PLAN_NODE_ID_SCHEMA_INDEX = DIFFERENCING_SCHEMA.index("left_child_plan_node_id")
RIGHT_CHILD_PLAN_NODE_ID_SCHEMA_INDEX = DIFFERENCING_SCHEMA.index("right_child_plan_node_id")
TARGET_START_SCHEMA_INDEX = DIFFERENCING_SCHEMA.index("total_cost")


class PlanDiffIncompleteSubinvocationException(Exception):
    """
    Exception is raised to indicate that differencing encountered a subinvocation with insufficient
    data. This can happen because certain OU events were dropped.
    """


class PlanDiffInvalidDataException(Exception):
    """
    Exception is raised to indicate that invalid data was encountered. Invalid data can mean that certain
    fields are corrupted or differencing produced invalid/inconsistent results.
    """


class PlanDiffUnsupportedParallelException(Exception):
    """
    Exception is raised to indicate that differencing encountered a subinvocation that indicates parallel
    query execution. As we currently don't support parallel query execution, this exception is raised.
    """
