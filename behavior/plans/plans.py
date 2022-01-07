from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame


class PlanTree:
    def __init__(self, query_id: str, json_plan: dict[str, Any]):
        self.query_id: str = query_id
        self.root: PlanNode = PlanNode(json_plan)
        self.parent_id_to_child_ids: dict[int, list[int]] = build_id_map(self.root)


class PlanNode:
    def __init__(self, json_plan: dict[str, Any]):
        self.node_type: str = json_plan["Node Type"]
        self.startup_cost: float = json_plan["Startup Cost"]
        self.total_cost: float = json_plan["Total Cost"]
        self.plan_node_id: int = json_plan["plan_node_id"]
        self.depth: int = json_plan["depth"]
        self.plans: list[PlanNode] = (
            [PlanNode(child_plan) for child_plan in json_plan["Plans"]]
            if "Plans" in json_plan
            else []
        )

    def __repr__(self) -> str:
        indent = ">" * (self.depth + 1)  # incremented so we always prefix with >
        return f"{indent} type: {self.node_type}, total_cost: {self.total_cost}, node_id: {self.plan_node_id}, depth: {self.depth}"


def build_id_map(root: PlanNode) -> dict[int, list[int]]:
    node_map: dict[int, list[int]] = {}
    node_map[root.plan_node_id] = [child.plan_node_id for child in root.plans]

    for child in root.plans:
        _build_id_map(child, node_map)

    return node_map


def _build_id_map(node: PlanNode, node_map: dict[int, list[int]]) -> None:
    node_map[node.plan_node_id] = [child.plan_node_id for child in node.plans]
    for child in node.plans:
        _build_id_map(child, node_map)


def show_plan_tree(plan_tree: PlanTree) -> None:
    logging.info("\n===== QueryID: %s =====", plan_tree.query_id)
    logging.info("Parent ID to Child IDs: %s", plan_tree.parent_id_to_child_ids)
    logging.info("%s", plan_tree.root)

    for child in plan_tree.root.plans:
        show_plan_node(child)


def show_plan_node(plan_node: PlanNode) -> None:
    logging.info("%s", plan_node)

    for child in plan_node.plans:
        show_plan_node(child)


def set_node_ids(json_plan: dict[str, Any]) -> None:
    json_plan["plan_node_id"] = 0
    json_plan["depth"] = 0
    next_node_id: int = 1

    if "Plans" in json_plan:
        for child in json_plan["Plans"]:
            next_node_id = _set_node_ids(child, next_node_id, 1)


def _set_node_ids(json_plan: dict[str, Any], next_node_id: int, depth: int) -> int:
    json_plan["plan_node_id"] = next_node_id
    json_plan["depth"] = depth
    next_node_id += 1

    if "Plans" in json_plan:
        for child in json_plan["Plans"]:
            next_node_id = _set_node_ids(child, next_node_id, depth + 1)

    return next_node_id


def get_plan_trees(
    raw_data_dir: Path, tscout_query_ids: set[str]
) -> dict[str, PlanTree]:
    plan_file_path: Path = raw_data_dir / "plan_file.csv"
    cols = ["queryid", "planid", "plan"]
    dtypes: dict[str, Any] = {"queryid": np.int64, "planid": int, "plan": str}
    plan: DataFrame = pd.read_csv(plan_file_path, usecols=cols, dtype=dtypes)
    plan["queryid"] = plan["queryid"].astype(np.uint64).astype(str)
    benchmark_plans = []

    query_id_to_plan_tree: dict[str, PlanTree] = {}

    for _, row in plan.iterrows():
        query_id: str = row["queryid"]

        if query_id in tscout_query_ids:
            json_plan: dict[str, Any] = json.loads(row["plan"])["Plan"]
            benchmark_plans.append(json_plan)
            set_node_ids(json_plan)
            query_id_to_plan_tree[query_id] = PlanTree(query_id, json_plan)

    outpath: Path = raw_data_dir / "differenced/LOG_bench_plans.txt"
    with outpath.open(mode="w") as f:
        for plan in benchmark_plans:
            f.write(json.dumps(plan))
            f.write("\n")

    return query_id_to_plan_tree
