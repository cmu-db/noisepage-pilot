from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame
from plumbum import cli

from behavior import (
    BASE_TARGET_COLS,
    BENCHDB_TO_TABLES,
    DIFF_COLS,
    LEAF_NODES,
    PLAN_NODE_NAMES,
)

COMMON_SCHEMA: list[str] = [
    "statement_id",
    "rid",
    "pid",
    "query_id",
    "plan_node_id",
    "left_child_plan_node_id",
    "right_child_plan_node_id",
    "statement_timestamp",
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
                    + [
                        "left_child_plan_node_id",
                        "right_child_plan_node_id",
                        "plan_node_id",
                    ]
                ):
                    assert not found, f"col: {init_col} and {common_col}"
                    assert init_col not in remapper, remapper
                    remapper[init_col] = common_col
                    found = True

        df = df.rename(columns=remapper)
        rids: list[str] = [uuid.uuid4().hex for _ in range(df.shape[0])]
        df["rid"] = rids
        df["ou_name"] = ou_name
        df["statement_id"] = (
            df["query_id"].astype(str) + "_" + df["statement_timestamp"].astype(str) + "_" + df["pid"].astype(str)
        )
        assert df.index.is_unique and df.index.size == df.shape[0]
        remapped[ou_name] = df

    return remapped


def load_tscout_data(tscout_data_dir: Path, logdir: Path) -> tuple[dict[str, DataFrame], DataFrame]:

    ou_to_df: dict[str, DataFrame] = {}

    # Phase 1: Create unified dataframe
    # Load all OU files into a dict of dataframes.
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

    # Remap the common columns into the common schema.
    ou_to_df = remap_cols(ou_to_df)

    unified: DataFrame = pd.concat([df[COMMON_SCHEMA] for df in ou_to_df.values()], axis=0)
    unified.sort_values(by=["statement_id", "start_time", "plan_node_id"], axis=0, inplace=True)

    # Partition multiple OU invocations within a single statement.
    invocation_ids = []
    invocation_id = 0
    prev_statement_id = None

    for _, row in unified.iterrows():
        statement_id = row["statement_id"]
        if statement_id != prev_statement_id or row["plan_node_id"] == 0:
            invocation_id += 1
            prev_statement_id = statement_id
        invocation_ids.append(invocation_id)

    unified["invocation_id"] = invocation_ids
    unified.to_csv(logdir / "unified_initial.csv", index=False)

    # Phase 2: Filter and log all incomplete query_ids
    unified.set_index("query_id", drop=False, inplace=True)
    query_id_to_plan_node_ids = {}
    for query_id in pd.unique(unified.index):
        temp = unified.loc[query_id]["plan_node_id"].values.tolist()
        plan_node_ids = set(temp)
        query_id_to_plan_node_ids[query_id] = plan_node_ids

    with (logdir / "query_plans.json").open("w", encoding="utf-8") as f:
        json_dict = json.dumps({str(k): list(v) for k, v in query_id_to_plan_node_ids.items()})
        f.write(json_dict)

    # All query plans must be numbered from 0 to NUM_PLAN_NODES - 1.
    # Verify this invariant and remove/log all plans not satisfying it.
    incomplete_query_ids = []
    for query_id, plan_node_ids in query_id_to_plan_node_ids.items():
        required_plan_node_ids = set(range(len(plan_node_ids)))

        if plan_node_ids != required_plan_node_ids:
            incomplete_query_ids.append(query_id)

    print(f"incomplete query_ids: {incomplete_query_ids}")
    # Log incomplete query identifiers.
    with (logdir / "incomplete_query_ids.csv").open("w", encoding="utf-8") as f:
        f.write("incomplete_query_ids\n")
        output = [f"{query_id}\n" for query_id in sorted(incomplete_query_ids)]
        f.writelines(output)

    # Remove all incomplete query_ids
    unified.loc[incomplete_query_ids].to_csv(logdir / "incomplete_query_id_data.csv", index=False)
    unified = unified[~unified.query_id.isin(incomplete_query_ids)]
    unified.to_csv(logdir / "unified_without_incomplete_query_ids.csv", index=False)

    # Phase 3: Filter and log all invocation_ids with an incorrect set of plan_node_ids.
    incomplete_invocation_ids = set()
    unified.set_index("invocation_id", drop=False, inplace=True)
    for invocation_id in pd.unique(unified.index):
        query_id = unified.loc[invocation_id]["query_id"]

        if isinstance(query_id, (DataFrame, pd.Series)):
            query_id = query_id.min()

        required_plan_node_ids = query_id_to_plan_node_ids[query_id]
        invocation_plan_node_ids = unified.loc[invocation_id]["plan_node_id"]

        if isinstance(invocation_plan_node_ids, (DataFrame, pd.Series)):
            invocation_plan_node_ids = set(invocation_plan_node_ids.values.tolist())
        elif isinstance(query_id, np.int64):
            invocation_plan_node_ids = set([invocation_plan_node_ids])

        if required_plan_node_ids != invocation_plan_node_ids:
            incomplete_invocation_ids.add(invocation_id)

    # Log incomplete invocation identifiers.
    with (logdir / "incomplete_invocation_ids.csv").open("w", encoding="utf-8") as f:
        f.write("incomplete_invocation_id\n")
        output = [f"{invocation_id}\n" for invocation_id in sorted(incomplete_invocation_ids)]
        f.writelines(output)

    # incomplete_invocations = unified.loc[incomplete_invocation_ids]
    incomplete_invocations = unified[unified.invocation_id.isin(incomplete_invocation_ids)]
    incomplete_invocations.to_csv(logdir / "incomplete_invocations.csv", index=False)
    unified = unified[~unified.invocation_id.isin(incomplete_invocation_ids)]
    unified.to_csv(logdir / "unified_without_incomplete_invocation_ids.csv", index=False)

    # Phase 4: Propagate changes back to original dataframes
    ou_to_df = {ou_name: df.set_index("rid", drop=False, inplace=False) for ou_name, df in ou_to_df.items()}
    incomplete_rids = incomplete_invocations["rid"]
    ou_to_df = {ou_name: df[~df.rid.isin(incomplete_rids)] for ou_name, df in ou_to_df.items()}

    for ou_name, df in ou_to_df.items():
        if df.shape[0] > 0:
            df.to_csv(logdir / f"filtered_{ou_name}.csv")
        else:
            logger.warning("OU: %s has no data after filtering", ou_name)

    return ou_to_df, unified


def diff_one_invocation(invocation: DataFrame) -> dict[str, NDArray[np.float64]]:
    rid_to_diffed_costs: dict[str, NDArray[np.float64]] = {}
    invocation.set_index("plan_node_id", drop=False, inplace=True)
    child_cols = ["left_child_plan_node_id", "right_child_plan_node_id"]
    assert (
        invocation["plan_node_id"].value_counts().max() == 1
    ), f"An invocation can only have one plan root!  Invocation Data: {invocation}"

    for _, parent_row in invocation.iterrows():
        parent_rid: str = parent_row["rid"]
        child_ids: NDArray[np.int64] = [id for id in parent_row[child_cols].values if id != -1]
        diffed_costs: NDArray[np.float64] = parent_row[DIFF_COLS].values

        for child_id in child_ids:
            child_row = invocation.loc[child_id]
            assert isinstance(child_row, pd.Series), f"Child row must always be a Pandas Series {child_row}"
            child_costs: NDArray[np.float64] = child_row[DIFF_COLS].values
            diffed_costs -= child_costs

        rid_to_diffed_costs[parent_rid] = diffed_costs

    return rid_to_diffed_costs


def diff_all_plans(unified: DataFrame, logdir: Path) -> DataFrame:

    all_query_ids: set[str] = set(pd.unique(unified["query_id"]))
    records: list[list[Any]] = []
    logger.info("Num query_ids: %s", len(all_query_ids))
    unified.to_csv(logdir / "unified_final_before_diffing.csv")
    unified.set_index("query_id", drop=False, inplace=True)

    for query_id in all_query_ids:
        query_invocations = unified.loc[query_id]
        if isinstance(query_invocations, pd.Series):
            continue
        assert isinstance(query_invocations, DataFrame)

        node_ids: pd.Series = query_invocations["plan_node_id"]
        node_counts: pd.Series = node_ids.value_counts()
        assert node_counts.min() == node_counts.max(), f"Invalid node_id set.  Node_counts: {node_counts}"

        assert (
            query_invocations["rid"].value_counts().max() == 1
        ), f"Found duplicate rids in query_invocations: {query_invocations}"

        query_invocation_ids = set(pd.unique(query_invocations["invocation_id"]))
        logger.info("Query ID: %s, Num invocations: %s", query_id, len(query_invocation_ids))
        indexed_invocations = query_invocations.set_index("invocation_id", drop=False, inplace=False)

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


def save_results(diff_data_dir: Path, ou_to_df: dict[str, DataFrame], diffed_cols: DataFrame) -> None:

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
            d.name for d in experiment_root.iterdir() if d.is_dir() and d.name in BENCHDB_TO_TABLES
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
