import datetime
from typing import Dict

import numpy as np
import pandas as pd
import sklearn.metrics.pairwise
import sklearn.neighbors
import sklearn.preprocessing
from plumbum import cli
from preprocessor import Preprocessor
from sklearn.cluster import DBSCAN


class Clusterer:
    """
    Cluster query templates based on the algorithms from QueryBot5000.

    [QueryBot5000]
    Lin Ma, Dana Van Aken, Ahmed Hefny, Gustavo Mezerhane, Andrew Pavlo,
    and Geoffrey J. Gordon. 2018. Query-based Workload Forecasting for
    Self-Driving Database Management Systems. SIGMOD 2018.

    Attributes
    ----------
    _df : pd.Dataframe
        Dataframe of counts grouped by (template, log_time_s)
        where log_time_s is aggregated to the clustering_interval
    n_samples : int
        Number of samples to use for calculating similarity between arrival rates.
    rho : float
        Similarity threshold used to determine template cluster membership.
    min_time : pd.Timestamp
        Earliest timestamp seen in _df.
    max_time : pd.Timestamp
        Latest timestamp seen in _df.
    cluster_interval : pd.Timedelta
        Time interval the df is aggregated by.
    n : int
        Number of datapoints in _df.
    cluster_gap : int
        Only use every x "time steps" to iterate for online clustering.
    n_gaps : int
        Number of time steps to to run online clustering.
    _dbgname : dict (string:int)
        Reverse lookup from query template string to an id.
    """

    def __init__(
        self,
        dataframe,
        n_samples=10000,
        rho=0.8,
        cluster_interval=pd.Timedelta(seconds=1),
    ):
        """
        Cluster the provided dataframe according to QueryBot5000.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe containing the query templates to be clustered.
        n_samples : int
            The number of timestamps to sample.
        rho : float
            Cosine similarity threshold for query template clustering.
        cluster_interval : pd.TimeDelta
            Time interval to group and count the query templates.
        """
        assert dataframe.index.names == ["query_template", "log_time_s"]
        assert dataframe.columns.values == ["count"]
        self._df = dataframe
        self.n_samples = n_samples
        self.rho = rho

        # Cluster interval of every second.
        self.min_time = self._get_timestamps().min()
        self.max_time = self._get_timestamps().max()

        self.interval_delta = cluster_interval
        self.n = int((self.max_time - self.min_time) / self.interval_delta + 1)

        self.cluster_gap = 1
        self.n_gaps = self.n // self.cluster_gap + 1

        # Represent query templates with integers for concise readability.
        self._dbgname = {
            template_str: template_id
            for template_id, template_str in dict(
                enumerate(self._get_queries())
            ).items()
        }

        # Cluster the queries.
        self.assignment_df = self._cluster_offline()

    def _get_queries(self):
        """
        Get the query templates being clustered.

        Returns
        -------
        queries : List[str]
            A list of the query templates being clustered.
        """
        return sorted(set(self._df.index.get_level_values(0)))

    def _get_timestamps(self):
        """
        Get all the timestamps across all the query templates.

        Returns
        -------
        timestamps : pd.DatetimeIndex
            All the timestamps.
        """

        # TODO(Mike): Are we ever relying on the date time index here to
        # reconstruct the time series with the clustering interval?
        # Could anything go wrong if this only has
        # 00:00, 00:01, 00:03, 00:04, but missing 00:02?
        return self._df.index.get_level_values(1)

    def _get_first_arrival(self, template):
        """
        Find the first arrival time for the given query.

        Parameters
        ----------
        template : str
            The query template to find the first arrival time for.

        Returns
        -------
        first_arrival : pd.Timestamp
            The first timestamp for the given query template.
        """
        return self._df.xs(template, level=0).index.min()

    @staticmethod
    def _query_df_range(df, template, start_time, end_time):
        """

        Parameters
        ----------
        df
        template
        start_time
        end_time

        Returns
        -------
        results : pd.DataFrame
        """
        # The first level can be dropped since query_template == template.
        return df.query(
            "`query_template` == @template"
            " and @start_time <= `log_time_s`"
            " and `log_time_s` < @end_time"
        ).droplevel(0)

    @staticmethod
    def _query_df(df, template, timestamps):
        """
        Get template counts, sampled by timestamps

        Parameters
        ----------
        df
        template
        timestamps

        Returns
        -------
        results : pd.DataFrame
        """
        # The first level can be dropped since query_template == template.
        df = df.query(
            "`query_template` == @template" " and `log_time_s` in @timestamps"
        ).droplevel(0)
        return df.reindex(timestamps, fill_value=0)

    @staticmethod
    def _query_series(series, timestamps):
        """
        Get values for a series, indexed by sample timestamps

        Parameters
        ----------
        series
        timestamps

        Returns
        -------
        results : pd.DataFrame
        """
        series = series.query("`log_time_s` in @timestamps")
        return series.reindex(timestamps, fill_value=0)

    @staticmethod
    def _similarity(s1, s2):
        """
        Compute the cosine similarity between the two series.
        Parameters
        ----------
        s1 : np.ndarray
        s2 : np.ndarray

        Returns
        -------
        similarity : np.float64
        """
        if s1.shape[0] == 0 or s2.shape[0] == 0:
            return 0
        # Reshape because we only have a single feature, the count.
        arr1 = s1.reshape(-1, 1)
        arr2 = s2.reshape(-1, 1)
        # Compute the cosine similarity.
        return sklearn.metrics.pairwise.cosine_similarity(arr1, arr2)[0][0]

    @staticmethod
    def _sample_timestamps(n, start_time, end_time, n_samples, interval):
        """

        Parameters
        ----------
        n : int
        start_time : pd.Timestamp
        end_time : pd.Timestamp
        n_samples : int
        interval : pd.TimeDelta

        Returns
        -------
        samples : pd.DatetimeArray
            Array of timestamps that were sampled.
        """
        if n > n_samples:
            offsets = np.random.choice(a=n, size=n_samples, replace=False)
        else:
            offsets = np.arange(n)
        timestamps = []
        for offset in offsets:
            next_time = start_time + interval * offset
            if next_time >= end_time:
                break
            timestamps.append(next_time)
        return pd.array(timestamps)

    @staticmethod
    def _build_neighbors(centers, timestamps, n_neighbors):
        """

        Parameters
        ----------
        centers
        timestamps
        n_neighbors

        Returns
        -------
        neighbors : sklearn.neighbors.NearestNeighbors | None
        """
        clusters = sorted(centers.keys())
        samples = np.array(
            [
                Clusterer._query_series(centers[cluster], timestamps).values
                for cluster in clusters
            ]
        )

        if len(samples) == 0:
            neighbors = None
        else:
            samples = samples.reshape(len(clusters), -1)
            normalized_samples = sklearn.preprocessing.normalize(samples, copy=False)
            neighbors = sklearn.neighbors.NearestNeighbors(
                n_neighbors=n_neighbors, algorithm="kd_tree", metric="l2"
            )
            neighbors.fit(normalized_samples)
        return neighbors

    def _modify_cluster(self, positive, cluster, template, start_time, end_time):
        """Add or remove a template from a cluster.

        Parameters
        ----------
        positive : bool
            True for add, False for remove.
        cluster : int
            The Cluster to modify.
        template : string
            Template to add to or remove from.
        start_time, end_time : pd.Timestamp
            Current time range considered
        """
        modify_method = (
            self.centers[cluster].add if positive else self.centers[cluster].sub
        )

        self.centers[cluster] = modify_method(
            self._query_df_range(self._df, template, start_time, end_time), fill_value=0
        )
        self.cluster_sizes[cluster] += 1 if positive else -1

    def _adjust_template(
        self, template, current_time, old_assignment, timestamps, neighbors
    ):
        """Adjust template cluster assignment at current time.

        Parameters
        ----------
        template : string
            The query template we need to update.
        current_time : pd.Timestamp
            Timestamp of the current clustering iteration.
        old_assignment : int
            Template's previous cluster assignment.
        timestamps : np.array(pd.Timestamp)
            Array of timestamps to sample from the centers fo similarity measurement.
        neighbors : sklearn.neighbors.NearestNeighbors
            Nearest neighbor learner containing all the cluster centers.

        Returns
        -------
        The updated cluster assignment to be added to self.assignments.
        """
        end_time = current_time + self.cluster_gap * self.interval_delta
        # Only consider the last 10 seconds.
        start_time = max(self.min_time, end_time - datetime.timedelta(seconds=10))

        # If template has not appeared at this point in time; assignment is still None.
        if (old_assignment is None) and (
            current_time <= self._get_first_arrival(template)
        ):
            return None
        if old_assignment is not None:
            # Template is the last member of the cluster.
            last_cluster_element = self.cluster_sizes[old_assignment] == 1
            # Template still belongs to its old cluster.
            still_belongs = (
                Clusterer._similarity(
                    self._query_df(self._df, template, timestamps).values,
                    self._query_series(self.centers[old_assignment], timestamps).values,
                )
                > self.rho
            )
            # If the template still belongs.
            if last_cluster_element or still_belongs:
                # reason = ('L' if last_cluster_element else '') + ('B' if still_belongs else '')
                # print(f'Template stayed in cluster {old_cluster} because ({reason}): {self._dbgname[template]}')
                return old_assignment

            # Otherwise, eliminate the template from its old cluster.
            self._modify_cluster(False, old_assignment, template, start_time, end_time)
            # print(f'Template eliminated from cluster {old_cluster}: {self._dbgname[template]}')

        new_assignment = None
        # Try to find a cluster membership for the template.
        if neighbors is None:
            for cluster in self.centers.keys():
                if (
                    self._similarity(
                        self._query_df(self._df, template, timestamps).values,
                        self._query_series(self.centers[cluster], timestamps).values,
                    )
                    > self.rho
                ):
                    new_assignment = cluster
                    break
        else:
            data = self._query_df(self._df, template, timestamps)[
                "count"
            ].values.reshape(1, -1)
            data = sklearn.preprocessing.normalize(data)
            neighbor = neighbors.kneighbors(data, return_distance=False)[0][0]
            clusters = sorted(self.centers.keys())
            if (
                self._similarity(data, self.centers[clusters[neighbor]].values)
                > self.rho
            ):
                new_assignment = clusters[neighbor]

        # If this template found a cluster to join, then make the assignment and continue.
        if new_assignment is not None:
            # description = 'joined' if old_assignment is None else 'reassigned to'
            # print(f'Template {description} cluster {new_cluster}: {self._dbgname[template]}')
            self._modify_cluster(True, new_assignment, template, start_time, end_time)
            return new_assignment

        # Otherwise, this template needs a new cluster. Make a new cluster.
        new_assignment = self.next_cluster
        self.next_cluster += 1

        self.centers[new_assignment] = self._query_df_range(
            self._df, template, start_time, end_time
        )
        assert self.centers[new_assignment].index.name == "log_time_s"
        assert self.centers[new_assignment].columns.values == ["count"]
        if self.centers[new_assignment].shape[0] == 0:
            print(
                f"WARNING: cluster {new_assignment} has no items."
                f"Does the following query appear within the lookback window:"
                f"{self._dbgname[template]}"
            )

        self.cluster_sizes[new_assignment] = 1
        self.cluster_totals[new_assignment] = 0
        print(
            f"Created cluster {new_assignment} based on template: {self._dbgname[template]}"
        )
        return new_assignment

    def _cluster_online(self):
        # Map cluster id to df representing center of cluster.
        self.centers: Dict[int, pd.DataFrame] = {}
        self.cluster_totals: Dict[int, int] = {}
        self.cluster_sizes: Dict[int, int] = {}

        # Array representing the assignment of template to clusters at a given time.
        self.assignments = [
            (
                self.min_time,
                {template: None for template in sorted(self._get_queries())},
            )
        ]

        # Begin at min time with no assignments.
        current_time = self.min_time

        # The next cluster id to use.
        self.next_cluster = 0

        for gap in range(self.n_gaps):
            # End time is the next interval.
            next_time = current_time + self.cluster_gap * self.interval_delta
            # Only consider the last 10 seconds.
            start_time = max(self.min_time, next_time - datetime.timedelta(seconds=10))
            # Timestamps to consider.
            timestamps = self._sample_timestamps(
                self.n, start_time, next_time, self.n_samples, self.interval_delta
            )

            # Get assignment dicts.
            last_assignment = self.assignments[-1][1]
            assignment = last_assignment.copy()

            # Update counts for all the assignments made in the past round.
            for template in last_assignment:
                old_assignment = last_assignment[template]
                if old_assignment is not None:
                    counts = self._query_df_range(
                        self._df, template, current_time, next_time
                    )
                    self.centers[old_assignment] = self.centers[old_assignment].add(
                        counts, fill_value=0
                    )
                    self.cluster_totals[old_assignment] += counts.sum().values[0]

            # If possible, build a kdtree of neighbors.
            neighbors = self._build_neighbors(self.centers, timestamps, n_neighbors=1)

            # For each template, try to assign a cluster.
            for template in self._get_queries():
                assignment[template] = self._adjust_template(
                    template=template,
                    current_time=current_time,
                    old_assignment=last_assignment[template],
                    timestamps=timestamps,
                    neighbors=neighbors,
                )

            # If possible, build an updated kdtree of neighbors. we need n_neighbors=2
            # because our query points are centers, so the second closest neighbor is the merge candidate.
            neighbors = self._build_neighbors(self.centers, timestamps, n_neighbors=2)
            root = [None] * len(self.centers)
            clusters = sorted(self.centers.keys())
            if len(clusters) > 1:
                # Try to merge clusters.
                for i, cluster in enumerate(clusters):
                    merge_cluster = None
                    data = self._query_series(self.centers[cluster], timestamps)[
                        "count"
                    ].values.reshape(1, -1)
                    data = sklearn.preprocessing.normalize(data)
                    neighbor = neighbors.kneighbors(data, return_distance=False)

                    neighbor_inds = neighbor[0]
                    if clusters[neighbor_inds[0]] == cluster:
                        neighbor = neighbor_inds[1]
                    else:
                        neighbor = neighbor_inds[0]
                    while root[neighbor] is not None:
                        neighbor = root[neighbor]
                    is_similar = (
                        self._similarity(
                            self._query_series(
                                self.centers[cluster], timestamps
                            ).values,
                            self._query_series(
                                self.centers[clusters[neighbor]], timestamps
                            ).values,
                        )
                        > self.rho
                    )
                    if cluster != clusters[neighbor] and is_similar:
                        merge_cluster = clusters[neighbor]
                    if merge_cluster is not None:
                        self.centers[merge_cluster] = self.centers[merge_cluster].add(
                            self.centers[cluster], fill_value=0
                        )
                        self.cluster_sizes[merge_cluster] += self.cluster_sizes[cluster]
                        del self.centers[cluster]
                        del self.cluster_sizes[cluster]
                        if neighbors is not None:
                            root[i] = neighbor
                        for template in self._get_queries():
                            if assignment[template] == cluster:
                                assignment[template] = merge_cluster
                                print(
                                    f"Template merged from cluster {cluster} into {merge_cluster}: "
                                    f"{self._dbgname[template]}"
                                )
            self.assignments.append((next_time, assignment))
            current_time = next_time
            for cluster, df in self.centers.items():
                if df.shape[0] == 0:
                    print(f"WARNING: gap {gap} cluster {cluster} has no items.")
        for template, cluster in self.assignments[-1][1].items():
            print(self._dbgname[template], "->", cluster)
        self.num_clusters = len(self.centers)

    def _cluster_offline(self):
        next_time = self.max_time + self.cluster_gap * self.interval_delta
        # TODO(Mike): only consider the last 10 seconds? or sample everything?
        start_time = self.min_time
        # Sample timestamps to consider.
        timestamps = self._sample_timestamps(
            self.n, start_time, next_time, self.n_samples, self.interval_delta
        )
        counts = np.array(
            [
                # Create (k,n) matrix where there are
                # k templates, n_sample features for DBSCAN.
                self._query_df(self._df, template, timestamps).values.reshape((-1))
                for template in self._get_queries()
            ]
        )

        clustering = DBSCAN(eps=1 - self.rho, metric="cosine", min_samples=1).fit(
            counts
        )
        labels = clustering.labels_
        reverse_lookup = {
            template_id: template_str
            for template_str, template_id in self._dbgname.items()
        }
        final_assignments = {
            reverse_lookup[template_id]: cluster_id
            for template_id, cluster_id in enumerate(labels)
        }
        return pd.DataFrame(
            final_assignments.items(), columns=["query_template", "cluster"]
        ).set_index("query_template")


class ClustererCLI(cli.Application):
    preprocessor_parquet = cli.SwitchAttr("--preprocessor-parquet", str, mandatory=True)
    output_parquet = cli.SwitchAttr("--output-parquet", str, mandatory=True)

    def main(self):
        print(f"Loading preprocessor data from {self.preprocessor_parquet}.")
        preprocessor = Preprocessor(parquet_path=self.preprocessor_parquet)

        # TODO(Mike): This should not be hardcoded, since many components
        # of the forecaster depend on this. Should be a shared constant somewhere.
        cluster_interval = pd.Timedelta(milliseconds=250)
        df = preprocessor.get_grouped_dataframe_interval(cluster_interval)
        df.index.rename(["query_template", "log_time_s"], inplace=1)
        print("Clustering query templates.")
        clusterer = Clusterer(df, cluster_interval=cluster_interval)
        print("Generating cluster assignments.")
        clusterer.assignment_df.to_parquet(self.output_parquet)
        print("Done!")


if __name__ == "__main__":
    ClustererCLI.run()
