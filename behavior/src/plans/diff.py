import os
import shutil
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame, Index
from tqdm import tqdm

from src import (
    BEHAVIOR_DATA_DIR,
    BENCHDB_TO_TABLES,
    COMMON_SCHEMA,
    DIFF_COLS,
    LEAF_NODES,
    PLAN_SCHEMA,
)
from src.plans.plans import PlanTree, get_plan_trees

# def verify_invocation_ids(unified: DataFrame) -> None:
#     inv_to_query_id: dict[int, str] = {}
#     inv_to_node_ids: dict[int, set[int]] = {}

#     df: DataFrame = unified[["query_id", "global_invocation_id", "plan_node_id"]].values.tolist()
#     for query_id, inv_id, node_id in df:

#         # verify each global_invocation_id maps to the same query_id
#         if inv_id in inv_to_query_id:
#             old_query_id = inv_to_query_id[inv_id]
#             assert (
#                 query_id == old_query_id
#             ), f"Found conflicting query_ids for inv_id: {inv_id}, new_query_id: {query_id}, old_query_id: {old_query_id}"
#         else:
#             inv_to_query_id[inv_id] = query_id

#         # verify each global_invocation_id has no duplicate plan_node_ids
#         if inv_id in inv_to_node_ids:
#             assert (
#                 node_id not in inv_to_node_ids[inv_id]
#             ), f"Found duplicate plan_node_id: {node_id} for inv_id: {inv_id}"
#             inv_to_node_ids[inv_id].add(node_id)
#         else:
#             inv_to_node_ids[inv_id] = {node_id}


# Unifying pg_store_plans and TScout Results:
# 1. Make sure every TScout query_id is in pg_store_plans results
# 2. Save relevant plans
# 3. Log and discard irrelevant plans


# Process the tscout data
# 1. Add invocation IDs for each query
# 2. Find, log, and filter all malformed invocations
def process_tscout_data(
    diff_data_dir: Path, unified: DataFrame, tscout_dfs: list[DataFrame]
) -> tuple[list[DataFrame], DataFrame]:

    unified = add_invocation_ids(diff_data_dir, unified)
    unified.to_csv(f"{diff_data_dir}/LOG_unified_with_invocations.csv", index=False)

    unified, tscout_dfs = filter_incomplete(diff_data_dir, unified, tscout_dfs)
    unified.to_csv(f"{diff_data_dir}/LOG_unified_filtered.csv", index=False)

    return tscout_dfs, unified


def load_tscout_data(tscout_data_dir: Path) -> tuple[list[DataFrame], DataFrame]:
    ou_to_df: dict[str, DataFrame] = {
        f.stem: pd.read_csv(f)
        for f in tscout_data_dir.glob("*.csv")
        if f.name.startswith("Exec") and os.stat(f).st_size > 0
    }
    tscout_dfs: list[DataFrame] = []

    for ou_name, df in ou_to_df.items():
        mapper: dict[str, str] = {}
        for col in df.columns:
            for mapper_value in PLAN_SCHEMA:
                if mapper_value in col:
                    mapper[col] = mapper_value
        df = df.rename(columns=mapper)
        rids: list[str] = [uuid.uuid4().hex for _ in range(df.shape[0])]
        df["rid"] = rids
        df["ou_name"] = ou_name
        df["query_id"] = df["query_id"].astype(str)
        tscout_dfs.append(df)

    unified: DataFrame = pd.concat([tdf[COMMON_SCHEMA] for tdf in tscout_dfs], axis=0)
    unified = unified.sort_values(by=["query_id", "start_time", "plan_node_id"], axis=0)
    diff_data_dir: Path = tscout_data_dir.parent / "differenced"
    unified.to_csv(f"{diff_data_dir}/LOG_unified_initial.csv", index=False)

    # we use a few different indexes for unified, starting with query_id
    unified.set_index("query_id", drop=False, inplace=True)

    # we only need RID for the tscout_df index
    tscout_dfs = [df.set_index("rid", drop=False, inplace=False) for df in tscout_dfs]

    for df in tscout_dfs:
        if df.shape[0] > 0:
            ou_name = df.iloc[0]["ou_name"]
            df.to_csv(f"{diff_data_dir}/LOG_{ou_name}_filtered.csv")
        else:
            print("no data for OU after filtering")

    return tscout_dfs, unified


def filter_incomplete(
    diff_data_dir: Path, unified: DataFrame, tscout_dfs: list[DataFrame]
) -> tuple[DataFrame, list[DataFrame]]:
    query_id_to_node_ids: defaultdict[str, set[int]] = defaultdict(set)
    inv_id_to_node_ids: defaultdict[tuple[str, int], set[int]] = defaultdict(set)

    for (_, row) in unified.iterrows():
        query_id: str = row["query_id"]
        inv_id: int = row["global_invocation_id"]
        node_id: int = row["plan_node_id"]

        query_id_to_node_ids[query_id].add(node_id)
        inv_id_to_node_ids[(query_id, inv_id)].add(node_id)

    broken_inv_ids: set[int] = set()

    for query_id, expected_ids in query_id_to_node_ids.items():
        matched_inv_ids: list[int] = [inv_id for (q_id2, inv_id) in inv_id_to_node_ids.keys() if q_id2 == query_id]

        for inv_id in matched_inv_ids:
            actual_ids: set[int] = inv_id_to_node_ids[(query_id, inv_id)]
            symdiff: set[int] = expected_ids.symmetric_difference(actual_ids)

            if len(symdiff) > 0:
                broken_inv_ids.add(inv_id)

    unified.set_index("global_invocation_id", drop=False, inplace=True)
    working_ids: Index = unified.index.difference(broken_inv_ids)
    unified.loc[broken_inv_ids].to_csv(f"{diff_data_dir}/LOG_broken_phase2.csv")
    filt_unified: DataFrame = unified.loc[working_ids]
    filt_unified.set_index("rid", drop=False, inplace=True)

    # apply filtering to all tscout dataframes
    rid_idx: Index = Index(data=filt_unified["rid"], dtype=str)
    filt_tscout_dfs: list[DataFrame] = [filter_by_rid(rid_idx, df) for df in tscout_dfs]

    return filt_unified, filt_tscout_dfs


# Invocation ID algorithm
# Each record has [rid, query_id, plan_node_id, start_time, end_time]
# We sort these by query_id, plan_node_id, start_time
# Then we scan them, incrementing counters each time we observe a
# terminating condition for an invocation.
def add_invocation_ids(diff_data_dir: Path, unified: DataFrame) -> DataFrame:
    unified.set_index("rid", drop=False, inplace=True)

    prev_query_id: int = 0
    root_end: int = 0
    query_invocation_id: int = 0
    global_invocation_id: int = 0
    query_invocation_ids: list[int] = []
    global_invocation_ids: list[int] = []
    broken_rids: list[str] = []
    inv_cols: list[str] = [
        "rid",
        "query_id",
        "plan_node_id",
        "left_child_plan_node_id",
        "right_child_plan_node_id",
        "start_time",
        "end_time",
    ]
    invocation_data: list[list[Any]] = unified[inv_cols].values.tolist()

    for rid, query_id, plan_node_id, curr_start, curr_end in invocation_data:
        if query_id != prev_query_id:
            root_end = curr_end
            query_invocation_id = 0
            prev_query_id = query_id
            global_invocation_id += 1
        elif plan_node_id == 0:
            root_end = curr_end
            query_invocation_id += 1
            global_invocation_id += 1
        elif curr_start > root_end:
            root_end = curr_end
            broken_rids.append(rid)
            global_invocation_id += 1

        query_invocation_ids.append(query_invocation_id)
        global_invocation_ids.append(global_invocation_id)

    assert len(query_invocation_ids) == len(unified.index)
    working_rids: Index = unified.index.difference(broken_rids)
    unified["query_invocation_id"] = query_invocation_ids
    unified["global_invocation_id"] = global_invocation_ids

    unified.to_csv(f"{diff_data_dir}/LOG_unified_before_filtering.csv", index=False)
    unified.loc[broken_rids].to_csv(f"{diff_data_dir}/LOG_broken_phase1.csv", index=False)

    unified = unified.loc[working_rids]

    # verify_invocation_ids(unified)

    return unified


def filter_by_rid(rid_idx: Index, df: DataFrame) -> DataFrame:
    df.set_index("rid", drop=False, inplace=True)
    filtered_idx = rid_idx.intersection(Index(data=df["rid"], dtype=str))
    return df.loc[filtered_idx]


def diff_one_plan(plan_tree: PlanTree, invocation: DataFrame) -> dict[str, NDArray[np.float64]]:
    rid_to_diffed_costs: dict[str, NDArray[np.float64]] = {}
    invocation.set_index("plan_node_id", drop=False, inplace=True)

    for parent_id, parent_row in invocation.iterrows():
        parent_rid: str = parent_row["rid"]
        child_ids: list[int] = plan_tree.parent_id_to_child_ids[parent_id]
        diffed_costs: NDArray[np.float64] = parent_row[DIFF_COLS].values

        for child_id in child_ids:
            child_costs: NDArray[np.float64] = invocation.loc[child_id][DIFF_COLS].values
            diffed_costs -= child_costs

        rid_to_diffed_costs[parent_rid] = diffed_costs

    return rid_to_diffed_costs


def save_results(diff_data_dir: Path, tscout_dfs: list[DataFrame], diffed_cols: DataFrame) -> None:

    diffed_cols.rename(columns=lambda col: f"diffed_{col}", inplace=True)

    # add the new columns onto the tscout dataframes
    for df in tscout_dfs:
        if df.shape[0] == 0:
            print("df has no records")
            continue

        ou_name = df.iloc[0]["ou_name"]
        df.drop(["ou_name", "rid"], axis=1, inplace=True)
        if ou_name in LEAF_NODES:
            df.to_csv(f"{diff_data_dir}/{ou_name}.csv", index=True)
            continue

        # find the intersection of RIDs between diffed_cols and each df
        rids_to_update = df.index.intersection(diffed_cols.index)
        print(f"num records to update: {rids_to_update.shape[0]}")

        if rids_to_update.shape[0] > 0:
            diffed_df = df.join(diffed_cols.loc[rids_to_update], how="inner")
            diffed_df.to_csv(f"{diff_data_dir}/{ou_name}.csv", index=True)
        else:
            diffed_df = df


def diff_all_plans(diff_data_dir: Path, unified: DataFrame) -> DataFrame:

    all_query_ids: set[str] = set(pd.unique(unified["query_id"]))
    query_id_to_plan_tree: dict[str, PlanTree] = get_plan_trees(diff_data_dir.parent, all_query_ids)
    records: list[list[Any]] = []

    print(f"Num query_ids: {len(all_query_ids)}")

    for query_id in tqdm(all_query_ids):
        plan_tree: PlanTree = query_id_to_plan_tree[query_id]

        if len(plan_tree.root.plans) > 0:
            query_invocations = unified.loc[query_id]
            if isinstance(query_invocations, pd.Series):
                continue
            assert isinstance(query_invocations, DataFrame)

            query_invocation_ids: set[int] = set(pd.unique(query_invocations["query_invocation_id"]))
            print(f"Query ID: {query_id}, Num invocations: {len(query_invocation_ids)}")
            query_invocations.set_index("query_invocation_id", drop=False, inplace=True)

            for invocation_id in query_invocation_ids:
                invocation = query_invocations.loc[invocation_id]
                if isinstance(invocation, pd.Series):
                    continue
                assert isinstance(invocation, DataFrame)

                for rid, diffed_costs in diff_one_plan(plan_tree, invocation).items():
                    assert isinstance(rid, str)
                    assert isinstance(diffed_costs, np.ndarray)
                    records.append([rid] + diffed_costs.tolist())

    diffed_cols = DataFrame(data=records, columns=["rid"] + DIFF_COLS)
    diffed_cols.to_csv(f"{diff_data_dir}/LOG_diffed_cols.csv", index=False)
    diffed_cols.set_index("rid", drop=True, inplace=True)

    return diffed_cols


def main(experiment: str) -> None:

    print(f"Differencing experiment: {experiment}")

    for mode in ["train", "eval"]:
        experiment_root: Path = BEHAVIOR_DATA_DIR / mode / experiment
        bench_names: list[str] = [
            d.name for d in experiment_root.iterdir() if d.is_dir() and d.name in BENCHDB_TO_TABLES
        ]

        # for bench_name in bench_names:
        for bench_name in ["tpcc"]:
            print(f"Mode: {mode} | Benchmark: {bench_name}")
            bench_root = experiment_root / bench_name
            tscout_data_dir = bench_root / "tscout"
            diff_data_dir: Path = bench_root / "differenced"
            if diff_data_dir.exists():
                shutil.rmtree(diff_data_dir)
            diff_data_dir.mkdir()

            tscout_dfs, unified = load_tscout_data(tscout_data_dir)
            diffed_cols: DataFrame = diff_all_plans(diff_data_dir, unified)
            save_results(diff_data_dir, tscout_dfs, diffed_cols)
