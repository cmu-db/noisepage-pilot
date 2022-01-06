import logging
from logging import Logger
from pathlib import Path

# postgres paths
PG_DIR = Path.home() / "postgres"
CMUDB_DIR = PG_DIR / "cmudb"
TSCOUT_DIR = CMUDB_DIR / "tscout"

# benchbase paths
BENCHBASE_DIR = Path.home() / "benchbase"
BENCHBASE_SNAPSHOT_DIR = BENCHBASE_DIR / "benchbase-2021-SNAPSHOT"
BENCHBASE_SNAPSHOT_PATH = BENCHBASE_DIR / "target" / "benchbase-2021-SNAPSHOT.zip"

# pilot paths
PILOT_DIR = Path.home() / "noisepage-pilot"
BEHAVIOR_DIR = PILOT_DIR / "behavior"

# config paths
CONFIG_ROOT = BEHAVIOR_DIR / "config"
MODEL_CONFIG_DIR = CONFIG_ROOT / "modeling"
DATAGEN_CONFIG_DIR = CONFIG_ROOT / "datagen"
PG_CONFIG_DIR = DATAGEN_CONFIG_DIR / "postgres"
BENCHBASE_CONFIG_DIR = DATAGEN_CONFIG_DIR / "benchbase"

# data paths
DATA_ROOT = BEHAVIOR_DIR / "data"
MODEL_DATA_DIR = DATA_ROOT / "models"
BEHAVIOR_DATA_DIR = DATA_ROOT / "training_data"
TRAIN_DATA_DIR = BEHAVIOR_DATA_DIR / "train"
EVAL_DATA_DIR = BEHAVIOR_DATA_DIR / "eval"

# other paths
SQLSMITH_DIR = Path.home() / "sqlsmith"
CLEANUP_SCRIPT_PATH = BEHAVIOR_DIR / "src/datagen/cleanup.py"

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

METHODS = [
    "lr",
    "huber",
    "svr",
    "kr",
    "rf",
    "gbm",
    "mlp",
    "mt_lasso",
    "lasso",
    "dt",
    "mt_elastic",
    "elastic",
]


LEAF_NODES: set[str] = {"ExecIndexScan", "ExecSeqScan", "ExecIndexOnlyScan", "ExecResult"}


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


def get_logger() -> Logger:

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    return logger
