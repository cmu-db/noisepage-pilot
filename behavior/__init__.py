from __future__ import annotations

from pathlib import Path

# Paths used throughout the behavior code
PILOT_DIR = Path(__file__).resolve().parent.parent

# Third-party
THIRD_PARTY_DIR = PILOT_DIR / "third-party"
PG_DIR = THIRD_PARTY_DIR / "postgres"
CMUDB_DIR = PG_DIR / "cmudb"
TSCOUT_DIR = CMUDB_DIR / "tscout"
BENCHBASE_DIR = THIRD_PARTY_DIR / "benchbase"
SQLSMITH_DIR = THIRD_PARTY_DIR / "sqlsmith"

# Configurations
CONFIG_DIR = PILOT_DIR / "config" / "behavior"
PG_CONFIG_DIR = CONFIG_DIR / "postgres"
BENCHBASE_CONFIG_DIR = CONFIG_DIR / "benchbase"

# Data
DATA_DIR = PILOT_DIR / "data" / "behavior"
MODEL_DATA_DIR = DATA_DIR / "models"
BEHAVIOR_DATA_DIR = DATA_DIR / "training_data"
TRAIN_DATA_DIR = BEHAVIOR_DATA_DIR / "train"
EVAL_DATA_DIR = BEHAVIOR_DATA_DIR / "eval"

# Logging
BEHAVIOR_LOG_DIR = PILOT_DIR / "log" / "behavior"

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

PLAN_NODE_NAMES = [
    "Agg",
    "Append",
    "CteScan",
    "CustomScan",
    "ForeignScan",
    "FunctionScan",
    "Gather",
    "GatherMerge",
    "Group",
    "HashJoinImpl",
    "IncrementalSort",
    "IndexOnlyScan",
    "IndexScan",
    "Limit",
    "LockRows",
    "Material",
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

LEAF_NODES: set[str] = {
    "IndexScan",
    "SeqScan",
    "IndexOnlyScan",
    "Result",
}

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


DIFF_COLS: list[str] = ["startup_cost", "total_cost"] + BASE_TARGET_COLS
