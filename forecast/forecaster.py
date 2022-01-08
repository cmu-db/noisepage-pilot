import csv
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from model import LSTM, ForecastDataset
from plumbum import cli
from preprocessor import Preprocessor


class ClusterForecaster:
    """
    Predict cluster amount in workload using trained LSTM
    """

    MODEL_PREFIX = "model_"

    @staticmethod
    def cluster_to_file(path, cluster):
        return f"{path}/{ClusterForecaster.MODEL_PREFIX}{cluster}.pkl"

    @staticmethod
    def get_cluster_from_file(filename):
        m = re.search(f"(?<={ClusterForecaster.MODEL_PREFIX})[^/]*(?=\\.pkl)", filename)
        if m is None:
            raise RuntimeError("Could not get cluster name")
        return m[0]

    def __init__(
        self,
        train_df: pd.DataFrame,
        cluster_interval: pd.Timedelta,
        prediction_seqlen: int,
        prediction_interval: pd.Timedelta,
        prediction_horizon: pd.Timedelta,
        save_path: str,
        top_k=5,
        override: bool = False,
    ):
        assert train_df.index.names[0] == "cluster"
        assert train_df.index.names[1] == "log_time_s"

        self.prediction_seqlen = prediction_seqlen
        self.prediction_interval = prediction_interval
        self.prediction_horizon = prediction_horizon
        self.models = {}

        model_files = glob.glob(str(Path(save_path) / f"{self.MODEL_PREFIX}*.pkl"))
        for filename in model_files:
            cluster_name = self.get_cluster_from_file(filename)
            self.models[int(cluster_name)] = LSTM.load(filename)
            print(f"loaded model for cluster {cluster_name}")

        print(f"Loaded {len(model_files)} models")

        if train_df is None:
            return

        # only consider top k clusters:
        cluster_totals = (
            train_df.groupby(level=0).sum().sort_values(by="count", ascending=False)
        )
        cluster_totals = cluster_totals / cluster_totals.sum()
        top_k = min(len(cluster_totals), top_k)
        train_df = train_df.loc[cluster_totals.index[:top_k], :]

        print("Training on cluster time series..")

        mintime = train_df.index.get_level_values(1).min()
        maxtime = train_df.index.get_level_values(1).max()

        dtindex = pd.date_range(
            start=mintime, end=maxtime, freq=cluster_interval, name="log_time_s"
        )
        labels = set(train_df.index.get_level_values(0).values)

        for cluster in labels:
            if cluster in self.models and not override:
                print(f"Already have model for cluster {cluster}, skipping")
                continue

            print(f"training model for cluster {cluster}")
            cluster_counts = (
                train_df[train_df.index.get_level_values(0) == cluster]
                .droplevel(0)
                .reindex(dtindex, fill_value=0)
            )

            self._train_cluster(cluster_counts, cluster, save_path)

    def _train_cluster(self, cluster_counts, cluster, save_path):
        dataset = ForecastDataset(
            cluster_counts,
            sequence_length=self.prediction_seqlen,
            horizon=self.prediction_horizon,
            interval=self.prediction_interval,
        )

        self.models[cluster] = LSTM(
            horizon=self.prediction_horizon,
            interval=self.prediction_interval,
            sequence_length=self.prediction_seqlen,
        )

        self.models[cluster].fit(dataset)
        self.models[cluster].save(self.cluster_to_file(save_path, cluster))

    def predict(self, cluster_df, cluster, start_time, end_time):
        """
        given a cluster dataset, attempt to return prediction of query count
        from a cluster within the given time-range
        """
        assert cluster_df.index.names[0] == "cluster"
        assert cluster_df.index.names[1] == "log_time_s"

        if cluster not in cluster_df.index.get_level_values(0):
            print(f"Could not find cluster {cluster} in cluster_df")
            return None

        cluster_counts = cluster_df[
            cluster_df.index.get_level_values(0) == cluster
        ].droplevel(0)

        # Truncate cluster_df to the time range necessary to generate
        # prediction range
        trunc_start = (
            start_time
            - self.prediction_horizon
            - (self.prediction_seqlen) * self.prediction_interval
        )
        trunc_end = end_time - self.prediction_horizon

        truncated = cluster_counts[
            (cluster_counts.index >= trunc_start) & (cluster_counts.index < trunc_end)
        ]

        dataset = ForecastDataset(
            truncated,
            sequence_length=self.prediction_seqlen,
            horizon=self.prediction_horizon,
            interval=self.prediction_interval,
        )

        # generate predictions
        predictions = [self.models[cluster].predict(seq) for seq, _ in dataset]

        # tag with timestamps
        pred_arr = [
            [dataset.get_y_timestamp(i), pred] for i, pred in enumerate(predictions)
        ]

        pred_df = pd.DataFrame(pred_arr, columns=["log_time_s", "count"])
        pred_df.set_index("log_time_s", inplace=True)
        return pred_df[start_time:]


class WorkloadGenerator:
    """
    Use preprocessed query template/params and cluster to generate
    representative workload
    """

    def __init__(self, preprocessor, assignment_df, cluster_interval):
        df = preprocessor.get_grouped_dataframe_interval(cluster_interval)
        df.index.rename(["query_template", "log_time_s"], inplace=True)

        # join to cluster and group by
        joined = df.join(assignment_df)

        # calculate weight of template within each cluster
        joined["cluster"].fillna(-1, inplace=True)
        summed = joined.groupby(["cluster", "query_template"]).sum()
        self._preprocessor = preprocessor
        self._percentages = summed / summed.groupby(level=0).sum()

    def get_workload(self, cluster, cluster_count):
        templates = self._percentages[
            self._percentages.index.get_level_values(0) == cluster
        ].droplevel(0)
        templates = templates * cluster_count

        # TODO(Mike): The true sample of parameters might be too inefficient,
        # But using the same parameters for all queries is not representative enough

        # True sample of parameters.
        # templates_with_param_vecs = [
        #     (template, self._preprocessor.sample_params(template, int(count)))
        #     for template, count in zip(templates.index.values, templates.values)
        # ]

        # Sample parameters once. Then use the same parameters
        # for all queries in the query template.
        templates_with_param_vecs = [
            (
                template,
                np.tile(
                    self._preprocessor.sample_params(template, 1)[0], (int(count), 1)
                ),
            )
            for template, count in zip(templates.index.values, templates.values)
        ]

        workload = [
            self._preprocessor.substitute_params(template, param_vec)
            for template, param_vecs in templates_with_param_vecs
            for param_vec in param_vecs
        ]
        workload = pd.DataFrame(workload, columns=["query"])
        predicted_queries = (
            workload.groupby("query").size().sort_values(ascending=False)
        )

        return predicted_queries


class ForecasterCLI(cli.Application):
    preprocessor_parquet = cli.SwitchAttr(
        ["-p", "--preprocessor-parquet"], str, mandatory=True
    )
    clusterer_parquet = cli.SwitchAttr(
        ["-c", "--clusterer-parquet"], str, mandatory=True
    )
    model_path = cli.SwitchAttr(["-m", "--model_path"], str, mandatory=True)

    start_ts = cli.SwitchAttr(["-s", "--start_time"], str, mandatory=True)
    end_ts = cli.SwitchAttr(["-e", "--end_time"], str, mandatory=True)

    output_csv = cli.SwitchAttr("--output_csv", str, mandatory=True)

    def main(self):
        print(f"Loading preprocessor data from {self.preprocessor_parquet}.")
        preprocessor = Preprocessor(parquet_path=self.preprocessor_parquet)
        cluster_interval = pd.Timedelta(seconds=5)
        df = preprocessor.get_grouped_dataframe_interval(cluster_interval)

        df.index.rename(["query_template", "log_time_s"], inplace=1)

        print("reading cluster assignments.")
        assignment_df = pd.read_parquet(self.clusterer_parquet)

        # join to cluster and group by
        joined = df.join(assignment_df)
        joined["cluster"].fillna(-1, inplace=True)
        clustered_df = joined.groupby(["cluster", "log_time_s"]).sum()

        # TODO(MIKE): check how many templates are not part of known
        # clusters (i.e. cluster = -1)

        pred_interval = pd.Timedelta(seconds=10)
        pred_horizon = pd.Timedelta(seconds=60)

        forecaster = ClusterForecaster(
            clustered_df,
            cluster_interval=cluster_interval,
            prediction_seqlen=10,
            prediction_interval=pred_interval,
            prediction_horizon=pred_horizon,
            save_path=self.model_path,
        )

        wg = WorkloadGenerator(preprocessor, assignment_df, cluster_interval)
        clusters = set(clustered_df.index.get_level_values(0).values)

        cluster_predictions = []
        for cluster in clusters:
            start_time = pd.Timestamp(self.start_ts)
            end_time = pd.Timestamp(self.end_ts)
            pred_df = forecaster.predict(clustered_df, cluster, start_time, end_time)
            prediction_count = pred_df["count"].sum()
            cluster_predictions.append(wg.get_workload(cluster, prediction_count))

        predicted_queries = pd.concat(cluster_predictions)
        predicted_queries.to_csv(self.output_csv, header=None, quoting=csv.QUOTE_ALL)


if __name__ == "__main__":
    ForecasterCLI.run()