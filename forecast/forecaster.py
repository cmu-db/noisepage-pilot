import csv
import datetime
from typing import Dict

import numpy as np
import pandas as pd

from plumbum import cli
from preprocessor import Preprocessor
from clusterer import Clusterer
from tqdm import tqdm, trange

class ClusterForecaster:
    """
    Predict cluster amount in workload using trained LSTM
    """
    def __init__(self):
        pass

class WorkloadGenerator:
    """
    Given information about a cluster, generate a representative workload
    of queries
    """
    
    def __init__(self, preprocessor, cluster_assignments):
        pass

class ForecasterCLI(cli.Application):
    preprocessor_parquet = cli.SwitchAttr("--preprocessor-parquet", str, mandatory=True)
    clusterer_parquet = cli.SwitchAttr("--clusterer-parquet", str, mandatory=True)
    output_csv = cli.SwitchAttr("--output_csv", str, mandatory=True)

    def main(self):
        print(f"Loading preprocessor data from {self.preprocessor_parquet}.")
        preprocessor = Preprocessor(parquet_path=self.preprocessor_parquet)
        clustering_interval = pd.Timedelta(seconds=5)
        df = preprocessor.get_grouped_dataframe_interval(clustering_interval)

        df.index.rename(["query_template", "log_time_s"], inplace=1)

        print("reading cluster assignments.")
        assignment_df = pd.read_parquet(self.clusterer_parquet)

        # join to cluster and group by
        clustered_df = df.join(assignment_df).groupby(["cluster", "log_time_s"]).sum()

        mintime = df.index.get_level_values(1).min()
        maxtime = df.index.get_level_values(1).max()
        dtindex = pd.date_range(start=mintime, end=maxtime,
                                freq=clustering_interval, name="log_time_s")
        labels = set(clustered_df.index.get_level_values(0).values)
        cluster_dict = {}
        for i in labels:
            cluster_counts = clustered_df[
                clustered_df.index.get_level_values(0) == i].droplevel(0).reindex(dtindex, fill_value=0)
            cluster_dict[i] = cluster_counts

        # TODO(WAN): The model is currently faked out here. @Mike
        templates = df.query(
            "`log_time_s` == '2021-12-26 13:32:55-05:00'"
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
        predicted_queries = (
            workload.groupby("query").size().sort_values(ascending=False)
        )

        predicted_queries.to_csv(self.output_csv, header=None, quoting=csv.QUOTE_ALL)


if __name__ == "__main__":
    ForecasterCLI.run()
