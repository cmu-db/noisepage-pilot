# noqa: E501 inspired by https://github.com/hyrise/index_selection_evaluation/blob/ca1dc87e20fe64f0ef962492597b77cd1916b828/selection/dbms/postgres_dbms.py
import constants
import logging
import psycopg


class Connector():
    def __init__(self):
        self._connection = psycopg.connect(dbname=constants.DB_NAME,
                                           user=constants.DB_USER,
                                           password=constants.DB_PASS,
                                           host=constants.DB_HOST)
        self._connection.autocommit = constants.AUTOCOMMIT
        logging.debug(
            f"Connected to {constants.DB_NAME} as {constants.DB_USER}")
        self.exec_commit_no_result("CREATE EXTENSION IF NOT EXISTS hypopg;")
        logging.debug("Enabled HypoPG")
        self.refresh_stats()

    def set_autocommit(self, autocommit: bool):
        self._connection.autocommit = autocommit

    def exec_commit_no_result(self, statement: str):
        self._connection.execute(statement)
        self._connection.commit()

    def exec_commit(self, statement: str) -> list[str]:
        cur = self._connection.execute(statement)
        results = cur.fetchall()
        self._connection.commit()
        return results

    def exec_transaction_no_result(self, statements: list[str]):
        with self._connection.transaction():
            cur = self._connection.cursor()
            for stmt in statements:
                cur.execute(stmt)

    def exec_transaction(self, statements: list[str]) -> list[list[str]]:
        res = []
        with self._connection.transaction():
            cur = self._connection.cursor()
            for stmt in statements:
                cur.execute(stmt)
                res.append(cur.fetchall())
        return res

    def close(self):
        self._connection.close()
        logging.debug(
            f"Disconnected from {constants.DB_NAME} as {constants.DB_USER}")

    # BEGIN: HypoPG operations on simulated indexes
    def simulate_index(self, create_stmt: str) -> int:
        hypopg_stmt = f"SELECT * FROM hypopg_create_index('{create_stmt}');"
        result = self.exec_commit(hypopg_stmt)
        return result[0][0]

    def drop_simulated_index(self, oid: int):
        hypopg_stmt = f"SELECT * FROM hypopg_drop_index({oid});"
        result = self.exec_commit(hypopg_stmt)
        assert(result[0][0] is True)

    def size_simulated_index(self, oid: int) -> int:
        hypopg_stmt = f"SELECT hypopg_relation_size({oid}) FROM hypopg_list_indexes;"
        result = self.exec_commit(hypopg_stmt)
        return result[0][0]
    # END

    # BEGIN: Postgres operations to simulate index changes
    def simulate_index_drop(self, ind_name: str):
        stmt = f"UPDATE pg_index SET indisvalid = false WHERE indexrelid = '{ind_name}'::regclass;"
        self.exec_commit_no_result(stmt)

    def undo_simulated_index_drop(self, ind_name: str):
        stmt = f"UPDATE pg_index SET indisvalid = true WHERE indexrelid = '{ind_name}'::regclass;"
        self.exec_commit_no_result(stmt)
    # END

    def get_plan(self, query: str) -> dict:
        stmt = f"EXPLAIN (format json) {query};"
        plan = self.exec_commit(stmt)[0][0][0]["Plan"]
        return plan

    def refresh_stats(self):
        self.exec_commit_no_result("ANALYZE;")

    # TODO: Consider removing restrictions on tables considered
    def get_table_info(self) -> dict[str, list[str]]:
        info = dict()
        tables = self.exec_commit(
            """
            SELECT relname
            FROM pg_class
            WHERE relkind='r' AND relname NOT LIKE 'pg_%' AND relname NOT LIKE 'sql_%';
            """
        )
        for table in tables:
            table = table[0]
            cols = self.exec_commit(
                f"SELECT column_name FROM information_schema.columns WHERE table_name='{table}';")
            for i in range(len(cols)):
                cols[i] = cols[i][0]
            info[table] = cols
        return info

    # TODO: Consider removing restrictions on indexes considered
    def get_index_info(self) -> list[(str, str, list[str], int, int)]:
        info = []
        indexes = self.exec_commit(
            """
            SELECT subq.relname as indexname, indexdef
            FROM pg_index
                JOIN (SELECT oid, relname
                        FROM pg_class
                        WHERE relname IN (SELECT indexname
                                          FROM   pg_indexes
                                          WHERE  schemaname = 'public')) AS subq
                ON indexrelid = subq.oid
                JOIN pg_indexes
                ON subq.relname = indexname
            WHERE  NOT ( indisunique OR indisprimary OR indisexclusion );
            """
        )
        for index in indexes:
            index_name = index[0]
            indexdef = index[1]
            table, cols = self._parse_index_info(indexdef)
            stats = self.exec_commit(
                f"""
                SELECT idx_scan, pg_relation_size(indexrelname::text)
                FROM pg_stat_all_indexes
                WHERE indexrelname='{index_name}';
                """
            )
            num_scans, size = stats[0]
            info.append((index_name, table, cols, num_scans, size))
        return info

    # TODO: Consider using sqlparse to parse this string
    def _parse_index_info(self, info: str) -> tuple[str, list[str]]:
        s1 = info.split(" USING ")
        assert(len(s1) == 2)
        s2 = s1[0].split(" ON ")
        assert(len(s2) == 2)
        s3 = s2[1].split('.')
        assert(len(s3) == 2)
        table = s3[1]
        s4 = s1[1]
        s5 = s4[s4.find("(") + 1:s4.find(")")]
        cols = s5.split(', ')
        return table, cols

    def get_unused_indexes(self):
        query = '''
        SELECT s.relname AS tablename,
            s.indexrelname AS indexname,
            pg_relation_size(s.indexrelid) AS index_size
        FROM pg_catalog.pg_stat_user_indexes s
        JOIN pg_catalog.pg_index i ON s.indexrelid = i.indexrelid
        WHERE s.idx_scan = 0      -- has never been scanned
            AND 0 <>ALL (i.indkey)  -- no index column is an expression
            AND s.schemaname = 'public'
            AND NOT i.indisunique   -- is not a UNIQUE index
            AND NOT EXISTS          -- does not enforce a constraint
                (SELECT 1 FROM pg_catalog.pg_constraint c
                WHERE c.conindid = s.indexrelid)
        ORDER BY pg_relation_size(s.indexrelid) DESC;
        '''
        return self.exec_commit(query)
