from __future__ import annotations

BENCHDB_TO_TABLES = {
    "tpcc": [
        "warehouse",
        "district",
        "customer",
        "item",
        "stock",
        "oorder",
        "history",
        "order_line",
        "new_order",
    ],
    "tatp": [
        "subscriber",
        "special_facility",
        "access_info",
        "call_forwarding",
    ],
    "tpch": [
        "region",
        "nation",
        "customer",
        "supplier",
        "part",
        "orders",
        "partsupp",
        "lineitem",
    ],
    "wikipedia": [
        "useracct",
        "watchlist",
        "ipblocks",
        "logging",
        "user_groups",
        "recentchanges",
        "page",
        "revision",
        "page_restrictions",
        "text",
    ],
    "voter": [
        "contestants",
        "votes",
        "area_code_state",
    ],
    "twitter": ["user_profiles", "tweets", "follows", "added_tweets", "followers"],
    "smallbank": ["accounts", "checking", "savings"],
    "sibench": ["sitest"],
    "seats": [
        "country",
        "airline",
        "airport",
        "customer",
        "flight",
        "airport_distance",
        "frequent_flyer",
        "reservation",
        "config_profile",
        "config_histograms",
    ],
    "resourcestresser": ["iotable", "cputable", "iotablesmallrow", "locktable"],
    "noop": ["fake"],
    "epinions": ["item", "review", "useracct", "trust", "review_rating"],
    "auctionmark": [
        "region",
        "useracct",
        "category",
        "config_profile",
        "global_attribute_group",
        "item",
        "item_comment",
        "useracct_feedback",
        "useracct_attributes",
        "item_bid",
        "useracct_watch",
        "global_attribute_value",
        "item_attribute",
        "item_image",
        "item_max_bid",
        "item_purchase",
        "useracct_item",
    ],
    "ycsb": ["usertable"],
}

# This list must be kept up to date with the OU definitions in cmu-db/postgres.
# OU_DEFS is defined in: https://github.com/cmu-db/postgres/blob/pg14/cmudb/tscout/model.py
PLAN_NODE_NAMES = [
    "Agg",
    "Append",
    "BitmapAnd",
    "BitmapHeapScan",
    "BitmapIndexScan",
    "BitmapOr",
    "CteScan",
    "CustomScan",
    "ForeignScan",
    "FunctionScan",
    "Gather",
    "GatherMerge",
    "Group",
    "Hash",
    "HashJoinImpl",
    "IncrementalSort",
    "IndexOnlyScan",
    "IndexScan",
    "Limit",
    "LockRows",
    "Material",
    "Memoize",
    "MergeAppend",
    "MergeJoin",
    "ModifyTable",
    "NamedTuplestoreScan",
    "NestLoop",
    "ProjectSet",
    "RecursiveUnion",
    "Result",
    "SampleScan",
    "SeqScan",
    "SetOp",
    "Sort",
    "SubPlan",
    "SubqueryScan",
    "TableFuncScan",
    "TidScan",
    "Unique",
    "ValuesScan",
    "WindowAgg",
    "WorkTableScan",
]

BASE_TARGET_COLS = [
    "cpu_cycles",
    "instructions",
    "cache_references",
    "cache_misses",
    "ref_cpu_cycles",
    "network_bytes_read",
    "network_bytes_written",
    "disk_bytes_read",
    "disk_bytes_written",
    "memory_bytes",
    "elapsed_us",
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


def standardize_input_data(df):
    """
    Standardizes input data for either model inference or for the data diff/model
    training pipeline. Function remaps non-target input columns that are suffixed
    by a column in STANDARDIZE_COLUMNS

    Parameters
    ----------
    df : pandas.DataFrame
        Input data dataframe that needs to be remapped.
        Remapping is done in-place.
    """

    # Determine which columns to remap. A non-target input column that is suffixed by a
    # column in STANDARDIZE_COLUMNS is remapped to produce a common schema.
    remapper: dict[str, str] = {}
    for init_col in df.columns:
        if init_col not in STANDARDIZE_COLUMNS and init_col not in BASE_TARGET_COLS:
            for common_col in STANDARDIZE_COLUMNS:
                if init_col.endswith(common_col):
                    remapper[init_col] = common_col
                    break
    df.rename(columns=remapper, inplace=True)
