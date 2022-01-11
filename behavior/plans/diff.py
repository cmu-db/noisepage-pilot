from __future__ import annotations

import logging
import os
import shutil
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame, Index
from plumbum import cli

from behavior import (
    BASE_TARGET_COLS,
    BENCHDB_TO_TABLES,
    DIFF_COLS,
    LEAF_NODES,
    PLAN_NODE_NAMES,
)

COMMON_SCHEMA: list[str] = [
    "rid",
    "pid",
    "query_id",
    "plan_node_id",
    "left_child_plan_node_id",
    "right_child_plan_node_id",
    "ou_name",
    "startup_cost",
    "total_cost",
    "plan_type",
    "plan_rows",
    "plan_width",
    "start_time",
    "end_time",
    "cpu_id",
] + BASE_TARGET_COLS

logger = logging.getLogger(__name__)


def verify_invocation_ids(unified: DataFrame) -> None:
    inv_to_query_id: dict[int, str] = {}
    inv_to_node_ids: dict[int, set[int]] = {}

    df: DataFrame = unified[
        ["query_id", "global_invocation_id", "plan_node_id"]
    ].values.tolist()
    for query_id, inv_id, node_id in df:

        # verify each global_invocation_id maps to the same query_id
        if inv_id in inv_to_query_id:
            old_query_id = inv_to_query_id[inv_id]
            assert (
                query_id == old_query_id
            ), f"Found conflicting query_ids for inv_id: {inv_id}, new_query_id: {query_id}, old_query_id: {old_query_id}"
        else:
            inv_to_query_id[inv_id] = query_id

        # verify each global_invocation_id has no duplicate plan_node_ids
        if inv_id in inv_to_node_ids:
            assert (
                node_id not in inv_to_node_ids[inv_id]
            ), f"Found duplicate plan_node_id: {node_id} for inv_id: {inv_id}"
            inv_to_node_ids[inv_id].add(node_id)
        else:
            inv_to_node_ids[inv_id] = {node_id}


def remap_cols(ou_to_df: dict[str, DataFrame]) -> dict[str, DataFrame]:
    remapped = {}
    for ou_name, df in ou_to_df.items():
        remapper: dict[str, str] = {}
        for init_col in df.columns:
            found = False
            for common_col in COMMON_SCHEMA:
                if (
                    common_col != init_col
                    and common_col in init_col
                    and init_col
                    not in BASE_TARGET_COLS
                    + ["left_child_plan_node_id", "right_child_plan_node_id", ""]
                ):
                    assert not found, f"col: {init_col} and {common_col}"
                    assert init_col not in remapper, remapper
                    remapper[init_col] = common_col
                    found = True

        df = df.rename(columns=remapper)
        rids: list[str] = [uuid.uuid4().hex for _ in range(df.shape[0])]
        df["rid"] = rids
        df["ou_name"] = ou_name
        df["query_id"] = df["query_id"].astype(str)
        df.set_index("rid", drop=False, inplace=True)
        remapped[ou_name] = df
        assert df.index.is_unique and df.index.size == df.shape[0]

    return remapped


def load_tscout_data(
    tscout_data_dir: Path, logdir: Path
) -> tuple[dict[str, DataFrame], DataFrame]:

    ou_to_df: dict[str, DataFrame] = {}

    for node_name in PLAN_NODE_NAMES:
        result_path = tscout_data_dir / f"Exec{node_name}.csv"

        if not result_path.exists():
            logger.error(
                "result doesn't exist for ou_name: %s, should be at path: %s",
                node_name,
                result_path,
            )
            sys.exit(1)

        if os.stat(result_path).st_size > 0:
            ou_to_df[node_name] = pd.read_csv(result_path)

    ou_to_df = remap_cols(ou_to_df)

    unified: DataFrame = pd.concat(
        [df[COMMON_SCHEMA] for df in ou_to_df.values()], axis=0
    )
    unified = unified.sort_values(by=["query_id", "start_time", "plan_node_id"], axis=0)
    unified.to_csv(logdir / "unified_initial.csv", index=False)

    # Process the tscout data
    # 1. Add invocation IDs for each query
    # 2. Find, log, and filter all malformed invocations
    unified = add_invocation_ids(unified, logdir)
    ou_to_df, unified = filter_incomplete(ou_to_df, unified, logdir)

    # we use a few different indexes for unified, starting with query_id
    unified.set_index("query_id", drop=False, inplace=True)

    ou_to_df = {
        ou_name: df.set_index("rid", drop=False, inplace=False)
        for ou_name, df in ou_to_df.items()
    }

    for ou_name, df in ou_to_df.items():
        if df.shape[0] > 0:
            df.to_csv(logdir / f"{ou_name}_filtered.csv")
        else:
            logger.warning("OU: %s has no data after filtering", ou_name)

    return ou_to_df, unified


def filter_incomplete(
    ou_to_df: dict[str, DataFrame],
    unified: DataFrame,
    logdir: Path,
) -> tuple[dict[str, DataFrame], DataFrame]:
    query_id_to_node_ids: defaultdict[str, set[int]] = defaultdict(set)
    inv_id_to_node_ids: defaultdict[tuple[str, int], set[int]] = defaultdict(set)

    assert unified["rid"].value_counts().max() == 1
    for (_, row) in unified.iterrows():
        query_id: str = row["query_id"]
        inv_id: int = row["global_invocation_id"]
        node_id: int = row["plan_node_id"]

        query_id_to_node_ids[query_id].add(node_id)
        inv_id_to_node_ids[(query_id, inv_id)].add(node_id)

    broken_inv_ids: set[int] = set()

    for query_id, expected_ids in query_id_to_node_ids.items():
        matched_inv_ids: list[int] = [
            inv_id for (q_id2, inv_id) in inv_id_to_node_ids.keys() if q_id2 == query_id
        ]

        for inv_id in matched_inv_ids:
            actual_ids: set[int] = inv_id_to_node_ids[(query_id, inv_id)]
            symdiff: set[int] = expected_ids.symmetric_difference(actual_ids)

            if len(symdiff) > 0:
                broken_inv_ids.add(inv_id)

    unified.set_index("global_invocation_id", drop=False, inplace=True)
    assert unified["rid"].value_counts().max() == 1
    working_ids: Index = unified.index.difference(broken_inv_ids).drop_duplicates()

    unified.loc[broken_inv_ids].to_csv(logdir / "broken_only_partial_plan.csv")
    assert not working_ids.has_duplicates
    filt_unified: DataFrame = unified.loc[working_ids]
    assert filt_unified["rid"].value_counts().max() == 1
    filt_unified.set_index("rid", drop=False, inplace=True)

    # apply filtering to all tscout dataframes
    rid_idx: Index = Index(data=filt_unified["rid"], dtype=str)
    filtered_ou_to_df = {
        ou_name: filter_by_rid(rid_idx, df) for ou_name, df in ou_to_df.items()
    }

    filt_unified.sort_values(by=["global_invocation_id"], axis=0).to_csv(
        logdir / "unified_filtered.csv", index=False
    )

    return filtered_ou_to_df, filt_unified


# Invocation ID algorithm
# Each record has [rid, query_id, plan_node_id, start_time, end_time]
# We sort these by query_id, plan_node_id, start_time
# Then we scan them, incrementing counters each time we observe a
# terminating condition for an invocation.
def add_invocation_ids(unified: DataFrame, logdir: Path) -> DataFrame:
    unified.set_index("rid", drop=False, inplace=True)

    prev_query_id: int = 0
    root_end: int = 0
    query_invocation_id: int = 0
    global_invocation_id: int = 0
    prev_pid = -1
    query_invocation_ids: list[int] = []
    global_invocation_ids: list[int] = []
    broken_rids: list[str] = []
    inv_cols: list[str] = [
        "rid",
        "pid",
        "query_id",
        "plan_node_id",
        "start_time",
        "end_time",
    ]

    invocation_data: list[list[Any]] = unified[inv_cols].values.tolist()

    for rid, pid, query_id, plan_node_id, curr_start, curr_end in invocation_data:
        if pid != prev_pid or query_id != prev_query_id:
            root_end = curr_end
            query_invocation_id = 0
            prev_query_id = query_id
            prev_pid = pid
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
    unified.to_csv(logdir / "unified_before_filtering.csv", index=False)
    unified.loc[broken_rids].to_csv(
        logdir / "cant_resolve_invocation_id.csv", index=False
    )
    unified = unified.loc[working_rids]
    unified.sort_values(by=["global_invocation_id", "plan_node_id"], axis=0).to_csv(
        logdir / "unified_with_invocations.csv", index=False
    )
    verify_invocation_ids(unified)

    return unified


def filter_by_rid(rid_idx: Index, df: DataFrame) -> DataFrame:
    df.set_index("rid", drop=False, inplace=True)
    filtered_idx = rid_idx.intersection(Index(data=df["rid"], dtype=str))
    return df.loc[filtered_idx]


def diff_one_invocation(invocation: DataFrame) -> dict[str, NDArray[np.float64]]:
    rid_to_diffed_costs: dict[str, NDArray[np.float64]] = {}
    invocation.set_index("plan_node_id", drop=False, inplace=True)
    assert (
        invocation["plan_node_id"].value_counts().max() == 1
    ), f"An invocation can only have one plan root!  Invocation Data: {invocation}"

    for _, parent_row in invocation.iterrows():
        parent_rid: str = parent_row["rid"]
        child_ids = [
            id
            for id in parent_row[
                ["left_child_plan_node_id", "right_child_plan_node_id"]
            ].values
            if id != -1
        ]
        diffed_costs: NDArray[np.float64] = parent_row[DIFF_COLS].values

        for child_id in child_ids:
            child_costs: NDArray[np.float64] = invocation.loc[child_id][
                DIFF_COLS
            ].values
            diffed_costs -= child_costs

        rid_to_diffed_costs[parent_rid] = diffed_costs

    return rid_to_diffed_costs


def diff_all_plans(unified: DataFrame, logdir: Path) -> DataFrame:

    all_query_ids: set[str] = set(pd.unique(unified["query_id"]))
    records: list[list[Any]] = []

    logger.info("Num query_ids: %s", len(all_query_ids))
    unified.to_csv(logdir / "final_unified_before_diffing.csv")

    for query_id in all_query_ids:
        query_invocations = unified.loc[query_id]
        node_ids: pd.Series = query_invocations["plan_node_id"]
        if isinstance(query_invocations, pd.Series):
            continue
        assert isinstance(query_invocations, DataFrame)

        node_counts: pd.Series = node_ids.value_counts()
        assert (
            node_counts.min() == node_counts.max()
        ), f"Invalid node_id set.  Node_counts: {node_counts}"

        assert (
            query_invocations["rid"].value_counts().max() == 1
        ), f"Found duplicate rids in query_invocations: {query_invocations}"

        query_invocation_ids: set[int] = set(
            pd.unique(query_invocations["query_invocation_id"])
        )

        logger.info(
            "Query ID: %s, Num invocations: %s", query_id, len(query_invocation_ids)
        )
        indexed_invocations = query_invocations.set_index(
            "query_invocation_id", drop=False, inplace=False
        )

        for invocation_id in query_invocation_ids:
            invocation = indexed_invocations.loc[invocation_id]
            if isinstance(invocation, pd.Series):
                continue
            assert isinstance(invocation, DataFrame)

            for rid, diffed_costs in diff_one_invocation(invocation).items():
                assert isinstance(rid, str)
                assert isinstance(diffed_costs, np.ndarray)
                records.append([rid] + diffed_costs.tolist())

    diffed_cols = DataFrame(data=records, columns=["rid"] + DIFF_COLS)
    diffed_cols.to_csv(logdir / "diffed_cols.csv", index=False)
    diffed_cols.set_index("rid", drop=True, inplace=True)

    return diffed_cols


def save_results(
    diff_data_dir: Path, ou_to_df: dict[str, DataFrame], diffed_cols: DataFrame
) -> None:

    diffed_cols.rename(columns=lambda col: f"diffed_{col}", inplace=True)

    # add the new columns onto the tscout dataframes
    for ou_name, df in ou_to_df.items():
        df.drop(["ou_name", "rid"], axis=1, inplace=True)
        if ou_name in LEAF_NODES:
            df.to_csv(f"{diff_data_dir}/{ou_name}.csv", index=True)
            continue

        # find the intersection of RIDs between diffed_cols and each df
        rids_to_update = df.index.intersection(diffed_cols.index)
        logger.info("Num records to update: %s", rids_to_update.shape[0])

        if rids_to_update.shape[0] > 0:
            diffed_df = df.join(diffed_cols.loc[rids_to_update], how="inner")
            diffed_df.to_csv(f"{diff_data_dir}/{ou_name}.csv", index=True)
        else:
            diffed_df = df


def main(data_dir, output_dir, experiment: str) -> None:
    logger.info("Differencing experiment: %s", experiment)

    for mode in ["train", "eval"]:
        experiment_root: Path = data_dir / mode / experiment
        logdir = output_dir / "log" / mode / experiment
        logdir.mkdir(parents=True, exist_ok=True)

        bench_names: list[str] = [
            d.name
            for d in experiment_root.iterdir()
            if d.is_dir() and d.name in BENCHDB_TO_TABLES
        ]

        for bench_name in bench_names:
            logger.info("Mode: %s | Benchmark: %s", mode, bench_name)
            bench_root = experiment_root / bench_name
            tscout_data_dir = bench_root / "tscout"
            diff_data_dir: Path = output_dir / "diff" / mode / experiment
            if diff_data_dir.exists():
                shutil.rmtree(diff_data_dir)
            diff_data_dir.mkdir(parents=True, exist_ok=True)

            tscout_dfs, unified = load_tscout_data(tscout_data_dir, logdir)
            diffed_cols: DataFrame = diff_all_plans(unified, logdir)
            save_results(diff_data_dir, tscout_dfs, diffed_cols)


class DataDiffCLI(cli.Application):
    dir_datagen_data = cli.SwitchAttr(
        "--dir-datagen-data",
        Path,
        mandatory=True,
        help="Directory containing DataGenerator output data.",
    )
    dir_output = cli.SwitchAttr(
        "--dir-output",
        Path,
        mandatory=True,
        help="Directory to output differenced CSV files to.",
    )

    def main(self):
        train_folder = self.dir_datagen_data / "train"
        experiments = sorted(path.name for path in train_folder.glob("*"))
        assert len(experiments) > 0, "No training data found?"
        latest_experiment = experiments[-1]
        main(self.dir_datagen_data, self.dir_output, latest_experiment)


if __name__ == "__main__":
    DataDiffCLI.run()
