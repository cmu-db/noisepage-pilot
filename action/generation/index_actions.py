from collections import defaultdict
import copy
import itertools
import numpy as np

from action import ActionGenerator
from action import Action
from connector import Connector
from workload import Workload

from pglast import ast, stream
from pglast.enums.parsenodes import *


class CreateIndexAction(Action):
    def __init__(self, table, cols, using=None):
        Action.__init__(self)
        self.table = table
        self.cols = cols
        self.using = using

    def index_name(self):
        colnames = [c.replace("_", "") for c in self.cols]
        return f'idx_{self.table}_{"_".join(colnames)}'

    def _to_sql(self):
        index_name = self.index_name()

        self.ast = ast.IndexStmt(
            idxname=index_name,
            relation=ast.RangeVar(relname=self.table, inh=True),
            accessMethod='btree' if self.using is None else self.using,
            indexParams=tuple(
                [
                    ast.IndexElem(
                        col,
                        ordering=SortByDir.SORTBY_DEFAULT,
                        nulls_ordering=SortByNulls.SORTBY_NULLS_DEFAULT,
                    ) for col in self.cols]
            ),
            idxcomment=None,
            if_not_exists=True,
        )
        return stream.RawStream(semicolon_after_last_statement=True)(self.ast)


class DropIndexAction(Action):
    def __init__(self, idxname, cascade=False):
        Action.__init__(self)
        self.idxname = idxname
        self.cascade = cascade

    def _to_sql(self):

        self.ast = ast.DropStmt(
            objects=[self.idxname],
            removeType=ObjectType.OBJECT_INDEX,
            behavior=DropBehavior.DROP_CASCADE if self.cascade else DropBehavior.DROP_RESTRICT,
            missing_ok=True,
        )
        return stream.RawStream(semicolon_after_last_statement=True)(self.ast)


class DropIndexGenerator(ActionGenerator):
    '''
    For each existing index, yield a DROP INDEX statement.
    '''

    def __init__(self, indexes) -> None:
        ActionGenerator.__init__(self)
        self.indexes = indexes

    def __iter__(self):
        for _, indname, _, _, _ in self.indexes:
            yield DropIndexAction(indname)


class SimpleIndexGenerator(ActionGenerator):
    '''
    For each query, this action generator produces a CREATE INDEX statement
    for each table's columns which appears together.
    '''

    # TODO: change this to take workload object instead of joint_refs
    def __init__(self, joint_refs) -> None:
        ActionGenerator.__init__(self)
        self.joint_refs = joint_refs

    def __iter__(self):
        for tablename, table in self.joint_refs.items():
            for cols in table:
                yield CreateIndexAction(tablename, cols)


class ExhaustiveIndexGenerator(ActionGenerator):
    '''
    For each query, this action generator produces a CREATE INDEX statement
    for each table's columns.
    '''

    def __init__(self, conn: Connector, max_width=1) -> None:
        ActionGenerator.__init__(self)
        table_info = conn.get_table_info()
        self.tables = list(table_info.keys())
        joint_refs = {k: defaultdict(lambda: defaultdict(np.uint64))
                      for k in self.tables}
        for (table, cols) in table_info:
            joint_refs[table][tuple(cols)] = 1
        self.joint_refs = joint_refs
        self.max_width = max_width

    def _iter_table_widths(self, table, width):
        col_perms = set()
        for cols in self.joint_refs[table]:
            for perm in itertools.permutations(cols, width):
                col_perms.add(perm)
                yield CreateIndexAction(table, perm)

    def __iter__(self) -> str:
        for table in self.tables:
            for width in range(1, self.max_width + 1):
                for action in self._iter_table_widths(table, width):
                    yield action


class WorkloadIndexGenerator(ActionGenerator):
    '''
    For each query, this action generator produces a CREATE INDEX statement
    for each table's columns which appears together.
    '''

    # TODO: change this to take workload object instead of joint_refs
    def __init__(self, workload: Workload, max_width=1) -> None:
        ActionGenerator.__init__(self)
        joint_refs = workload.get_where_colrefs()
        self.tables = list(joint_refs.keys())
        self.joint_refs = joint_refs
        self.actions_per_table = [len(joint_refs[t]) for t in self.tables]
        # Prefix sum of actions per table
        self.max_width = max_width

    def _iter_table_widths(self, table, width):
        col_perms = set()
        for cols in self.joint_refs[table]:
            for perm in itertools.permutations(cols, width):
                if perm in col_perms:
                    continue
                col_perms.add(perm)
                yield CreateIndexAction(table, perm)

    def __iter__(self) -> str:
        for table in self.tables:
            for width in range(1, self.max_width + 1):
                for action in self._iter_table_widths(table, width):
                    yield action


class TypedIndexGenerator(ActionGenerator):
    '''
    For each query, this action generator produces a CREATE INDEX statement
    for each table's columns which appears together.
    '''

    def __init__(self, upstream: ActionGenerator) -> None:
        ActionGenerator.__init__(self)
        self.upstream = upstream

    def __iter__(self) -> str:
        for orig_action in self.upstream:
            for using in ["HASH", "BRIN"]:
                new_action = copy.deepcopy(orig_action)
                new_action.using = using
                yield new_action
