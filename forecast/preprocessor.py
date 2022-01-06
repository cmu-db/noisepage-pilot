import glob
import re
import time
import warnings
from pathlib import Path
from typing import List

import pandas as pd
import pglast
from pandarallel import pandarallel
from plumbum import cli
from tqdm.contrib.concurrent import process_map

# Enable parallel pandas operations.
# pandarallel is a little buggy. For example, progress_bar=True does not work,
# and if you are using PyCharm you will want to enable "Emulate terminal in
# output console" instead of using the PyCharm Python Console.
# The reason we're using this library anyway is that:
# - The parallelization is dead simple: change .blah() to .parallel_blah().
# - swifter has poor string perf; we're mainly performing string ops.
# - That said, Wan welcomes any switch that works.
pandarallel.initialize(verbose=1)


class Preprocessor:
    """
    Convert PostgreSQL query logs into pandas DataFrame objects.
    """

    # The columns that constitute a CSVLOG file, as defined by PostgreSQL.
    # See: https://www.postgresql.org/docs/14/runtime-config-logging.html
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

    def get_dataframe(self):
        """
        Get a raw dataframe of query log data.

        Returns
        -------
        df : pd.DataFrame
            Dataframe containing the query log data.
            Note that irrelevant query log entries are still included.
        """
        return self._df

    def get_grouped_dataframe_interval(self, interval):
        """
        Get the pre-grouped version of query log data.

        Parameters
        ----------
        interval : pd.TimeDelta
            time interval to group and count the query templates

        Returns
        -------
        grouped_df : pd.DataFrame
            Dataframe containing the pre-grouped query log data.
            Grouped on query template and log time.
        """
        gb = self._df.groupby("query_template").resample(interval).size()
        grouped_df = pd.DataFrame(gb, columns=["count"])
        grouped_df.drop("", axis=0, level=0, inplace=True)
        return grouped_df

    def get_grouped_dataframe_params(self):
        """
        Get the pre-grouped version of query log data.

        Returns
        -------
        grouped_df : pd.DataFrame
            Dataframe containing the pre-grouped query log data.
            Grouped on query template and query parameters.
        """
        return self._grouped_df_params

    def get_params(self, query):
        """
        Find the parameters associated with a particular query.

        Parameters
        ----------
        query : str
            The query template to look up parameters for.

        Returns
        -------
        params : pd.Series
            The counts of parameters associated with a particular query.
            Unfortunately, due to quirks of the PostgreSQL CSVLOG format,
            the types of parameters are unreliable and may be stringly typed.
        """
        params = self._grouped_df_params.query("query_template == @query")
        return params.droplevel(0).squeeze(axis=1)

    def sample_params(self, query, n, replace=True, weights=True):
        """
        Find a sampling of parameters associated with a particular query.

        Parameters
        ----------
        query : str
            The query template to look up parameters for.
        n : int
            The number of parameter vectors to sample.
        replace : bool
            True if the sampling should be done with replacement.
        weights : bool
            True if the sampling should use the counts as weights.
            False if the sampling should be equal probability weighting.

        Returns
        -------
        params : np.ndarray
            Sample of the parameters associated with a particular query.
        """
        params = self.get_params(query)
        weight_vec = params if weights else None
        sample = params.sample(n, replace=replace, weights=weight_vec)
        return sample.index.to_numpy()

    @staticmethod
    def substitute_params(query_template, params):
        assert type(query_template) == str
        query = query_template
        keys = [f"${i}" for i in range(1, len(params) + 1)]
        for k, v in zip(keys, params):
            query = query.replace(k, v)
        return query

    @staticmethod
    def _read_csv(csvlog):
        """
        Read a PostgreSQL CSVLOG file into a pandas DataFrame.

        Parameters
        ----------
        csvlog : str
            Path to a CSVLOG file generated by PostgreSQL.

        Returns
        -------
        df : pd.DataFrame
            DataFrame containing the relevant columns for query forecasting.
        """
        # This function must have a separate non-local binding from _read_df
        # so that it can be pickled for multiprocessing purposes.
        return pd.read_csv(
            csvlog,
            names=Preprocessor._PG_LOG_COLUMNS,
            parse_dates=["log_time", "session_start_time"],
            usecols=[
                "log_time",
                "session_start_time",
                "command_tag",
                "message",
                "detail",
            ],
            header=None,
            index_col=False,
        )

    @staticmethod
    def _read_df(csvlogs):
        """
        Read the provided PostgreSQL CSVLOG files into a single DataFrame.

        Parameters
        ----------
        csvlogs : List[str]
            List of paths to CSVLOG files generated by PostgreSQL.

        Returns
        -------
        df : pd.DataFrame
            DataFrame containing the relevant columns for query forecasting.
        """
        return pd.concat(process_map(Preprocessor._read_csv, csvlogs))

    @staticmethod
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
        print("TODO(WAN): Disabled SQL format for being too slow.")
        # Prettify each SQL query for standardized formatting.
        # query = query.parallel_map(pglast.prettify, na_action='ignore')
        # Replace NA values (irrelevant log messages) with empty strings.
        query.fillna("", inplace=True)
        return query.astype(str)

    @staticmethod
    def _extract_params(detail_series):
        """
        Extract SQL parameters from the CSVLOG's detail column.
        If there are no such parameters, an empty {} is returned.

        Parameters
        ----------
        detail_series : pd.Series
            A series corresponding to the detail column of a CSVLOG file.

        Returns
        -------
        params : pd.Series
            A dict-typed series containing the parameters from the log.
        """

        def extract(detail):
            detail = str(detail)
            prefix = "parameters: "
            idx = detail.find(prefix)
            if idx == -1:
                return {}
            parameter_list = detail[idx + len(prefix) :]
            params = {}
            for pstr in parameter_list.split(", "):
                pnum, pval = pstr.split(" = ")
                assert pnum.startswith("$")
                assert pnum[1:].isdigit()
                params[pnum] = pval
            return params

        return detail_series.parallel_apply(extract)

    @staticmethod
    def _substitute_params(df, query_col, params_col):
        """
        Substitute parameters into the query, wherever possible.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe of query log data.
        query_col : str
            Name of the query column produced by _extract_query.
        params_col : str
            Name of the parameter column produced by _extract_params.
        Returns
        -------
        query_subst : pd.Series
            A str-typed series containing the query with parameters inlined.
        """

        def substitute(query, params):
            for k, v in params.items():
                query = query.replace(k, v)
            return query

        def subst(row):
            return substitute(row[query_col], row[params_col])

        return df.parallel_apply(subst, axis=1)

    @staticmethod
    def _parse(query_series):
        """
        Parse the SQL query to extract (prepared queries, parameters).

        Parameters
        ----------
        query_series : pd.Series
            SQL queries with the parameters inlined.

        Returns
        -------
        queries_and_params : pd.Series
            A series containing tuples of (prepared SQL query, parameters).
        """

        def parse(sql):
            new_sql, params, last_end = [], [], 0
            for token in pglast.parser.scan(sql):
                token_str = str(sql[token.start : token.end + 1])
                if token.start > last_end:
                    new_sql.append(" ")
                if token.name in ["ICONST", "FCONST", "SCONST"]:
                    # Integer, float, or string constant.
                    new_sql.append("$" + str(len(params) + 1))
                    params.append(token_str)
                else:
                    new_sql.append(token_str)
                last_end = token.end + 1
            new_sql = "".join(new_sql)
            return new_sql, tuple(params)

        return query_series.parallel_apply(parse)

    def _from_csvlogs(self, csvlogs):
        """
        Glue code for initializing the Preprocessor from CSVLOGs.

        Parameters
        ----------
        csvlogs : List[str]
            List of PostgreSQL CSVLOG files.

        Returns
        -------
        df : pd.DataFrame
            A dataframe representing the query log.
        """
        time_end, time_start = None, time.perf_counter()

        def clock(label):
            nonlocal time_end, time_start
            time_end = time.perf_counter()
            print("\r{}: {:.2f} s".format(label, time_end - time_start))
            time_start = time_end

        df = self._read_df(csvlogs)
        clock("Read dataframe")

        print("Extract queries: ", end="", flush=True)
        df["query_raw"] = self._extract_query(df["message"])
        df.drop(columns=["message"], inplace=True)
        clock("Extract queries")

        print("Extract parameters: ", end="", flush=True)
        df["params"] = self._extract_params(df["detail"])
        df.drop(columns=["detail"], inplace=True)
        clock("Extract parameters")

        print("Substitute parameters into query: ", end="", flush=True)
        df["query_subst"] = self._substitute_params(df, "query_raw", "params")
        df.drop(columns=["query_raw", "params"], inplace=True)
        clock("Substitute parameters into query")

        print("Parse query: ", end="", flush=True)
        parsed = self._parse(df["query_subst"])
        df[["query_template", "query_params"]] = pd.DataFrame(
            parsed.tolist(), index=df.index
        )
        clock("Parse query")

        # only keep the relevant columns for storage
        return df[["log_time", "query_template", "query_params"]]

    def __init__(self, csvlogs=None, parquet_path=None):
        """
        Initialize the preprocessor with either CSVLOGs or a HDF dataframe.

        Parameters
        ----------
        csvlogs : List[str] | None
            List of PostgreSQL CSVLOG files.

        hdf_path : str | None
            Path to a .h5 file containing a Preprocessor's get_dataframe().
        """
        if csvlogs is not None:
            df = self._from_csvlogs(csvlogs)
        else:
            assert parquet_path is not None
            df = pd.read_parquet(parquet_path)
            # convert params from array back to tuple so it is hashable
            df["query_params"] = df["query_params"].map(lambda x: tuple(x))

        # grouping queries by template-parameters count.
        gbp = df.groupby(["query_template", "query_params"]).size()
        grouped_by_params = pd.DataFrame(gbp, columns=["count"])
        # grouped_by_params.drop('', axis=0, level=0, inplace=True)
        # TODO(WAN): I am not sure if I'm wrong or pandas is wrong.
        #  Above raises ValueError: Must pass non-zero number of levels/codes.
        #  So we'll do this instead...
        grouped_by_params = grouped_by_params[~grouped_by_params.index.isin([("", ())])]
        self._df = df
        self._df.set_index("log_time", inplace=True)
        self._grouped_df_params = grouped_by_params


class PreprocessorCLI(cli.Application):
    query_log_folder = cli.SwitchAttr("--query-log-folder", str, mandatory=True)
    output_parquet = cli.SwitchAttr("--output-parquet", str, mandatory=True)

    def main(self):
        pgfiles = glob.glob(str(Path(self.query_log_folder) / "postgresql*.csv"))
        assert (
            len(pgfiles) > 0
        ), f"No PostgreSQL query log files found in: {self.query_log_folder}"
        preprocessor = Preprocessor(pgfiles)
        # TODO(WAN): The mixing of types in a column leads to
        #  a PerformanceWarning for PyTables. Feel free to fix.
        warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
        print("storing parquet")
        preprocessor.get_dataframe().to_parquet(self.output_parquet, compression="gzip")


if __name__ == "__main__":
    PreprocessorCLI.run()
