import re
from typing import List

import pandas as pd
import pglast
import connector
from pglast import ast, visitors
from collections import defaultdict
import numpy as np

_PG_LOG_COLUMNS: List[str] = [
    "log_time",
    "user_name",
    "database_name",
    "process_id",
    "connection_from",
    "session_id",
    "session_line_num",
    "command_tag",
    "session_start_time",
    "virtual_transaction_id",
    "transaction_id",
    "error_severity",
    "sql_state_code",
    "message",
    "detail",
    "hint",
    "internal_query",
    "internal_query_pos",
    "context",
    "query",
    "query_pos",
    "location",
    "application_name",
    "backend_type",
]


def _extract_query(message_series):
    """
    Extract SQL queries from the CSVLOG's message column.

    Parameters
    ----------
    message_series : pd.Series
        A series corresponding to the message column of a CSVLOG file.

    Returns
    -------
    query : pd.Series
        A str-typed series containing the queries from the log.
    """
    simple = r"statement: ((?:DELETE|INSERT|SELECT|UPDATE).*)"
    extended = r"execute .+: ((?:DELETE|INSERT|SELECT|UPDATE).*)"
    regex = f"(?:{simple})|(?:{extended})"
    query = message_series.str.extract(regex, flags=re.IGNORECASE)
    # Combine the capture groups for simple and extended query protocol.
    query = query[0].fillna(query[1])
    query.fillna("", inplace=True)
    return query.astype(str)


def parse_csv_log(file):
    """
    Extract queries from workload csv file and return df with fingerprint
    and corresponding queries
    """
    df = pd.read_csv(
        file, names=_PG_LOG_COLUMNS,
        parse_dates=["log_time", "session_start_time"],
        usecols=[
            "log_time",
            "session_start_time",
            "command_tag",
            "message",
            "detail",
        ],
        header=None,
        index_col=False)

    # filter out empty messages
    df = df[df["message"] != ""]
    df['detail'].fillna("", inplace=True)
    # extract queries and toss commits, sets, etc.
    df['queries'] = _extract_query(df['message'])
    df = df[df['queries'] != ""]
    df['fingerprint'] = df['queries'].apply(pglast.parser.fingerprint)
    return df[['fingerprint', 'queries']]


def find_colrefs(node: pglast.node.Node):
    """
    Find all column refs by scanning through a pglast node
    """
    if node is pglast.Missing:
        return []
    colrefs = []
    for subnode in node.traverse():
        if type(subnode) is pglast.node.Scalar:
            continue
        if type(subnode.ast_node) is ast.ColumnRef:
            colref = tuple([
                n.val.value for n in subnode.fields
                if type(n.ast_node) == ast.String])
            if len(colref) > 0:
                colrefs.append(colref)
    return colrefs


def get_all_colrefs(sql, table_cols):
    """
    Get all column refs from a sql statement which appear in
    WHERE and GROUP BYs

    Attempt to resolve aliases for table refs
    """
    tree = pglast.parse_sql(sql)

    aliases = {}
    where_colrefs = []
    group_by_colrefs = []
    referenced_tables = visitors.referenced_relations(tree)

    # mine the AST for aliases and colrefs
    for node in pglast.node.Node(tree).traverse():
        if type(node) is pglast.node.Scalar:
            continue
        if 'whereClause' in node.attribute_names:
            where_colrefs += find_colrefs(node.whereClause)
        if 'groupClause' in node.attribute_names:
            group_by_colrefs += find_colrefs(node.groupClause)
        if 'alias' in node.attribute_names and 'relname' in node.attribute_names:
            if node.alias is pglast.Missing or node.relname is pglast.Missing:
                continue
            if node.alias.aliasname.value in aliases:
                print("UH OH, double alias")
            aliases[node.alias.aliasname.value] = node.relname.value
    # resolve aliases and figure out actual table col refs
    where_potentials = parse_colref_aliases(
        where_colrefs, aliases, referenced_tables, table_cols)
    group_by_potentials = parse_colref_aliases(
        group_by_colrefs, aliases, referenced_tables, table_cols)
    return where_potentials, group_by_potentials


# parse_colref_aliases returns set of colrefs with resolved table names
def parse_colref_aliases(raw_colrefs, aliases, referenced_tables, table_cols):
    potential_colrefs = []
    for c in raw_colrefs:
        potential_tables = []
        p_col = None

        # The colref does not specify table, assume this col could be in
        # any of the referenced tables
        if len(c) == 1:
            p_col = c[0]
            potential_tables = [
                t for t in referenced_tables
                if t in table_cols and
                p_col in table_cols[t]]

        # The colref does specify table, also attempt to resolve alias
        if len(c) == 2:
            t = c[0]
            p_col = c[1]
            if t not in table_cols and t in aliases:
                t = aliases[t]
            potential_tables = [t]

        # Only add the table,col pair if it exists in the schema
        potential_colrefs += [
            (p_t, p_col) for p_t in potential_tables if (p_t in table_cols and p_col in table_cols[p_t])]
    return set(potential_colrefs)


def aggregate_templates(df, conn, percent_threshold=1):
    """
    Aggregate queries into templates based on pglast
    Only retain most common queries up to {percent_threshold} of the workload
    """

    table_cols = conn.get_table_info()

    aggregated = df[['queries', 'fingerprint']]\
        .groupby('fingerprint')\
        .agg([pd.DataFrame.sample, "count"])['queries']\
        .sort_values('count', ascending=False)
    aggregated['fraction'] = aggregated['count'] / aggregated['count'].sum()
    aggregated['cumsum'] = aggregated['fraction'].cumsum()
    filtered = pd.DataFrame(
        aggregated[aggregated['cumsum'] <= percent_threshold])

    # get column refs
    filtered['where_colrefs'], filtered['group_by_colrefs'] = zip(*filtered['sample'].apply(
        get_all_colrefs, args=(table_cols,)))

    return filtered[['sample', 'count', 'cumsum', 'where_colrefs', 'group_by_colrefs']]


def get_workload_colrefs(filtered):
    # TODO: all colref_types extraction are hard coded
    colref_types = ['where_colrefs', 'group_by_colrefs']
    table_colrefs_joint_counts = {k: defaultdict(lambda: defaultdict(np.uint64)) for k in colref_types}

    for colref_type_str in colref_types:
        for _, row in filtered.iterrows():
            refs = row[colref_type_str]
            tables = set([tab for (tab, _) in refs])
            for table in tables:
                cols_for_tabs = [col for (tab, col) in refs if tab == table]
                if len(cols_for_tabs) == 0:
                    continue
                joint_ref = tuple(dict.fromkeys(cols_for_tabs))
                table_colrefs_joint_counts[colref_type_str][table][joint_ref] += row['count']

    return table_colrefs_joint_counts
