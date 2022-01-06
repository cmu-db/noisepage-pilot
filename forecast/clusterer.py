import csv
import datetime
from typing import Dict

import numpy as np
import pandas as pd
import sklearn.metrics.pairwise
import sklearn.neighbors
import sklearn.preprocessing
from plumbum import cli
from preprocessor import Preprocessor
from tqdm import tqdm, trange


class Clusterer:
    """
    Cluster query templates based on the algorithms from QueryBot5000.

    [QueryBot5000]
    Lin Ma, Dana Van Aken, Ahmed Hefny, Gustavo Mezerhane, Andrew Pavlo,
    and Geoffrey J. Gordon. 2018. Query-based Workload Forecasting for
    Self-Driving Database Management Systems. SIGMOD 2018.
    """

    def __init__(self, dataframe, n_samples=10000, rho=0.8):
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
        """
        assert dataframe.index.names == ["query_template", "log_time_s"]
        assert dataframe.columns.values == ["count"]
        self._df = dataframe
        self.n_samples = n_samples
        self.rho = rho

        # Cluster interval of every second.
        self.min_time = self._get_timestamps().min()
        self.max_time = self._get_timestamps().max()
        self.n = (self.max_time - self.min_time).days * 24 * 60 * 60 + (self.max_time - self.min_time).seconds + 1
        self.cluster_gap = 1
        self.n_gaps = self.n // self.cluster_gap + 1

        # Represent query templates with integers for concise readability.
        self._dbgname = {v: k for k, v in dict(enumerate(self._get_queries())).items()}

        # Cluster the queries.
        self._cluster()

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
            "`query_template` == @template" " and @start_time <= `log_time_s`" " and `log_time_s` < @end_time"
        ).droplevel(0)

    @staticmethod
    def _query_df(df, template, timestamps):
        """

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
        df = df.query("`query_template` == @template" " and `log_time_s` in @timestamps").droplevel(0)
        return df.reindex(timestamps, fill_value=0)

    @staticmethod
    def _query_series(series, timestamps):
        """

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
    def _sample_timestamps(n, start_time, end_time, n_samples):
        """

        Parameters
        ----------
        n
        start_time
        end_time
        n_samples

        Returns
        -------
        samples : pd.DatetimeArray
            Array of timestamps that were sampled.
        """
        if n > n_samples:
            offsets = np.random.choice(a=n, size=n_samples, replace=False)
        else:
            offsets = np.arange(n)
        timestamps = [start_time]
        for offset in offsets:
            next_time = pd.Timedelta(seconds=offset) + start_time
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
        samples = np.array([Clusterer._query_series(centers[cluster], timestamps).values for cluster in clusters])

        if len(samples) == 0:
            neighbors = None
        else:
            samples = samples.reshape(len(clusters), -1)
            normalized_samples = sklearn.preprocessing.normalize(samples, copy=False)
            neighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors, algorithm="kd_tree", metric="l2")
            neighbors.fit(normalized_samples)
        return neighbors

    def _cluster(self):
        rho = self.rho

        centers: Dict[int, pd.DataFrame] = {}
        cluster_totals: Dict[int, int] = {}
        cluster_sizes: Dict[int, int] = {}

        assignments = [
            (
                self.min_time,
                {template: None for template in sorted(self._get_queries())},
            )
        ]

        current_time = self.min_time
        next_cluster = 0

        for gap in trange(self.n_gaps):
            next_time = current_time + datetime.timedelta(seconds=self.cluster_gap)
            # Up to last 10 seconds.
            start_time = max(self.min_time, next_time - datetime.timedelta(seconds=10))
            timestamps = Clusterer._sample_timestamps(self.n, start_time, next_time, self.n_samples)

            last_assignment = assignments[-1][1]
            assignment = assignments[-1][1].copy()

            # Update counts for all the assignments made in the past round.
            for template in last_assignment:
                old_cluster = last_assignment[template]
                if old_cluster is not None:
                    counts = Clusterer._query_df_range(self._df, template, current_time, next_time)
                    centers[old_cluster] = centers[old_cluster].add(counts, fill_value=0)
                    cluster_totals[old_cluster] += counts.sum().values[0]

            # If possible, build a kdtree of neighbors.
            neighbors = Clusterer._build_neighbors(centers, timestamps, n_neighbors=1)

            # For each template, try to assign a cluster.
            for template in self._get_queries():
                old_cluster = assignment[template]

                if old_cluster is not None:
                    # Test if the template still belongs to its old cluster.
                    last_cluster_element = cluster_sizes[old_cluster] == 1
                    still_belongs = (
                        Clusterer._similarity(
                            Clusterer._query_df(self._df, template, timestamps).values,
                            Clusterer._query_series(centers[old_cluster], timestamps).values,
                        )
                        > rho
                    )
                    # If the template still belongs, continue.
                    if last_cluster_element or still_belongs:
                        reason = ""
                        if last_cluster_element:
                            reason += "L"
                        if still_belongs:
                            reason += "B"
                        # print(
                        #     f"Template stayed in cluster {old_cluster}"
                        #     f"because ({reason}): {self._dbgname[template]}"
                        # )
                        continue
                    # Otherwise, eliminate the template from its old cluster.
                    cluster_sizes[old_cluster] -= 1
                    centers[old_cluster] = centers[old_cluster].sub(
                        Clusterer._query_df_range(self._df, template, start_time, next_time),
                        fill_value=0,
                    )
                    # print(f'Template eliminated from cluster {old_cluster}:'
                    #       f' {self._dbgname[template]}')

                # Test if template has appeared at this point in time; else, continue.
                if assignment[template] is None:
                    first_arrival = self._get_first_arrival(template)
                    if current_time <= first_arrival:
                        # print(
                        #     f"Template has not yet arrived at "
                        #     f"{current_time}, skipping: "
                        #     f"{self._dbgname[template]}"
                        # )
                        continue
                    # print(f'Template arrived at {current_time}: '
                    #       f'{self._dbgname[template]}')

                new_cluster = None
                # Try to assign to existing cluster.
                if neighbors is None:
                    for cluster in centers.keys():
                        if (
                            Clusterer._similarity(
                                self._query_df(self._df, template, timestamps).values,
                                self._query_series(centers[cluster], timestamps).values,
                            )
                            > rho
                        ):
                            new_cluster = cluster
                            break
                else:
                    data = Clusterer._query_df(self._df, template, timestamps)
                    data = data["count"].values.reshape(1, -1)
                    data = sklearn.preprocessing.normalize(data)
                    neighbor = neighbors.kneighbors(data, return_distance=False)[0][0]
                    clusters = sorted(centers.keys())
                    similarity = Clusterer._similarity(data, centers[clusters[neighbor]].values)
                    if similarity > rho:
                        new_cluster = clusters[neighbor]

                # If this template found a cluster to join,
                # then make the assignment and continue.
                if new_cluster is not None:
                    # description = (
                    #     "joined" if assignment[template] is None
                    #     else "reassigned to"
                    # )
                    # print(f'Template {description} '
                    #       f'cluster {new_cluster}: {self._dbgname[template]}')
                    assignment[template] = new_cluster
                    centers[new_cluster] = centers[new_cluster].add(
                        self._query_df_range(self._df, template, start_time, next_time),
                        fill_value=0,
                    )
                    cluster_sizes[new_cluster] += 1
                    continue

                # Otherwise, this template needs a new cluster. Make a new cluster.
                assignment[template] = next_cluster
                centers[next_cluster] = self._query_df_range(self._df, template, start_time, next_time)
                assert centers[next_cluster].index.name == "log_time_s"
                assert centers[next_cluster].columns.values == ["count"]
                # if centers[next_cluster].shape[0] == 0:
                #     print(f'WARNING: cluster {next_cluster} has no items. '
                #           f'Does the following query appear within the'
                #           f' lookback window: {self._dbgname[template]}')

                cluster_sizes[next_cluster] = 1
                cluster_totals[next_cluster] = 0
                # print(f'Created cluster {next_cluster} based on'
                #       f' template: {self._dbgname[template]}')
                # Update the cluster counter.
                next_cluster += 1

            root = [None] * len(centers)
            # If possible, build an updated kdtree of neighbors.
            neighbors = Clusterer._build_neighbors(centers, timestamps, n_neighbors=2)

            clusters = sorted(centers.keys())
            if len(clusters) > 1:
                # Try to merge clusters.
                for i, cluster in enumerate(clusters):
                    merge_cluster = None
                    data = Clusterer._query_series(centers[cluster], timestamps)["count"].values.reshape(1, -1)
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
                            self._query_series(centers[cluster], timestamps).values,
                            self._query_series(centers[clusters[neighbor]], timestamps).values,
                        )
                        > rho
                    )
                    if cluster != clusters[neighbor] and is_similar:
                        merge_cluster = clusters[neighbor]
                    if merge_cluster is not None:
                        centers[merge_cluster] = centers[merge_cluster].add(centers[cluster], fill_value=0)
                        cluster_sizes[merge_cluster] += cluster_sizes[cluster]
                        del centers[cluster]
                        del cluster_sizes[cluster]
                        if neighbors is not None:
                            root[i] = neighbor
                        for template in self._get_queries():
                            if assignment[template] == cluster:
                                assignment[template] = merge_cluster
                                # print(f'Template merged from cluster '
                                #       f'{cluster} into {merge_cluster}: '
                                #       f'{self._dbgname[template]}')
            assignments.append((next_time, assignment))
            current_time = next_time
            # empty_clusters = set(
            #     cluster for cluster, df in centers.items() if df.shape[0] == 0
            # )
            # if len(empty_clusters) > 0:
            #    print(f'WARNING: gap {gap} has empty clusters: {clusters}')

        self.assignments = assignments
        self.centers = centers
        self.cluster_totals = cluster_totals
        self.cluster_sizes = cluster_sizes
        self.num_clusters = len(self.centers)


class OnlineClusters:
    def __init__(self, clusterer):
        self._generate(clusterer)

    def _generate(self, clusterer):
        query_cum = clusterer._df.groupby(level=0).cumsum()
        MAX_CLUSTER_NUM = 5
        top_clusters = []
        coverage_lists = [[] for i in range(MAX_CLUSTER_NUM)]

        min_ts = clusterer._df.index.get_level_values(1).min()
        last_ts = min_ts
        online_clusters = {}

        for current_ts, assignment in tqdm(clusterer.assignments):
            cluster_totals = {}
            ts_total = 0

            for template, cluster in assignment.items():
                if cluster is None:
                    continue

                query = query_cum.query("`query_template` == @template and `log_time_s` <= @current_ts")
                max_ts = query.index.get_level_values(1).max()

                template_total = 0
                if (current_ts - max_ts).seconds < 24 * 60 * 60:
                    template_total = query.max().values[0]

                ts_total += template_total
                cluster_totals[cluster] = cluster_totals.get(cluster, 0) + template_total

            if len(cluster_totals) == 0:
                last_ts = current_ts
                continue

            sorted_clusters = sorted(cluster_totals.items(), key=lambda x: x[1], reverse=True)
            sorted_names, sorted_totals = zip(*sorted_clusters)

            lookahead = datetime.timedelta(seconds=10)

            current_top_clusters = sorted_clusters[:MAX_CLUSTER_NUM]
            for current_cluster, num_queries in current_top_clusters:
                if current_cluster not in online_clusters:
                    online_clusters[current_cluster] = {}
                    for template, cluster in assignment.items():
                        if current_cluster != cluster:
                            continue
                        start_ts = min_ts
                        end_ts = last_ts + lookahead
                        query = clusterer._df.query(
                            "`query_template` == @template"
                            " and @start_ts <= `log_time_s`"
                            " and `log_time_s` < @end_ts"
                        )
                        online_clusters[cluster] = query

            current_top_cluster_names = [cluster for cluster, _ in current_top_clusters]
            for template, cluster in assignment.items():
                if cluster not in current_top_cluster_names:
                    continue
                start_ts = last_ts + lookahead  # noqa
                end_ts = current_ts + lookahead  # noqa
                query = clusterer._df.query(
                    "`query_template` == @template" " and @start_ts <= `log_time_s`" " and `log_time_s` < @end_ts"
                )
                online_clusters[cluster] = online_clusters[cluster].add(query, fill_value=0)

            top_clusters.append((current_ts, current_top_clusters))
            for i in range(MAX_CLUSTER_NUM):
                coverage_lists[i].append(sum(sorted_totals[: i + 1] / ts_total))
            last_ts = current_ts

        coverage = [sum(cl) / len(cl) for cl in coverage_lists]

        trajs = {cluster: df.swaplevel().sum(level=0) for cluster, df in online_clusters.items()}

        self.top_clusters = top_clusters
        self.coverage = coverage
        self.online_clusters = online_clusters
        self.trajs = trajs


class ClustererCLI(cli.Application):
    preprocessor_hdf = cli.SwitchAttr("--preprocessor-hdf", str, mandatory=True)
    output_csv = cli.SwitchAttr("--output-csv", str, mandatory=True)

    def main(self):
        print(f"Loading preprocessor data from {self.preprocessor_hdf}.")
        preprocessor = Preprocessor(hdf_path=self.preprocessor_hdf)
        df = preprocessor.get_grouped_dataframe_seconds()
        print("Clustering query templates.")
        clusterer = Clusterer(df)
        print("Generating online clusters.")
        oc = OnlineClusters(clusterer)

        # TODO(WAN): The model is currently faked out here. @Mike
        templates = pd.concat(
            oc.online_clusters[cluster].query("`log_time_s` == '2021-12-06 14:23:47-05:00'")
            for cluster in oc.online_clusters
        )
        templates = templates.droplevel(1)

        # True sample of parameters.
        templates_with_param_vecs = [
            (template, preprocessor.sample_params(template, int(count)))
            for template, count in zip(templates.index.values, templates.values)
        ]
        # Sample parameters once. Then use the same parameters
        # for all queries in the query template.
        templates_with_param_vecs = [
            (
                template,
                np.tile(preprocessor.sample_params(template, 1)[0], (int(count), 1)),
            )
            for template, count in zip(templates.index.values, templates.values)
        ]
        workload = [
            preprocessor.substitute_params(template, param_vec)
            for template, param_vecs in templates_with_param_vecs
            for param_vec in param_vecs
        ]
        workload = pd.DataFrame(workload, columns=["query"])
        predicted_queries = workload.groupby("query").size().sort_values(ascending=False)

        predicted_queries.to_csv(self.output_csv, header=None, quoting=csv.QUOTE_ALL)


if __name__ == "__main__":
    ClustererCLI.run()
