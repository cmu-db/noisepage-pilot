from collections import defaultdict
import pandas as pd
import numpy as np
import pglast

import logparsing
import index_actions

import constants
import connector


# TODO: fold into workload obj


def get_workload_colrefs(filtered):
    # indexes = db_connector.get_existing_indexes()
    table_colrefs_joint_counts = defaultdict(lambda: defaultdict(np.uint64))

    for _, row in filtered.iterrows():
        refs = row['colrefs']
        tables = set([tab for (tab, _) in refs])
        for table in tables:
            cols_for_tabs = [col for (tab, col) in refs if tab == table]
            if len(cols_for_tabs) == 0:
                continue
            joint_ref = tuple(dict.fromkeys(cols_for_tabs))
            table_colrefs_joint_counts[table][joint_ref] += row['count']

    return table_colrefs_joint_counts


def find_indexes(obj):
    if type(obj) is list:
        return sum([find_indexes(x) for x in obj], [])
    if type(obj) is dict:
        res = sum([find_indexes(obj[x]) for x in obj], [])
        res += [obj['Index Name']] if 'Index Name' in obj else []
        return res
    return []


def generate_indexes(workload_csv):
    conn = connector.Connector()

    # TODO: fold these things into a workload obj
    parsed = logparsing.parse_csv_log(workload_csv)
    filtered = logparsing.aggregate_templates(parsed, conn)
    colrefs = get_workload_colrefs(filtered)
    # end: folded into wkld obj
    # where_colrefs = workload.get_where_colrefs() returns the same format as this "colrefs"
    exhaustive = index_actions.ExhaustiveIndexGenerator(
        colrefs, constants.MAX_IND_WIDTH)
    actions = list(exhaustive)
    return actions


if __name__ == "__main__":
    generate_indexes(constants.WORKLOAD_CSV)
