import logging

from plumbum import cli

from behavior.datagen import generate_workloads
from behavior.microservice import app
from behavior.modeling import train
from behavior.plans import diff

logger = logging.getLogger(__name__)


class BehaviorCLI(cli.Application):
    def main(self) -> None:
        pass


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s:%(asctime)s %(message)s", level=logging.INFO)
    BehaviorCLI.subcommand("generate_workloads", generate_workloads.GenerateWorkloadsCLI)
    BehaviorCLI.subcommand("datadiff", diff.DataDiffCLI)
    BehaviorCLI.subcommand("train", train.TrainCLI)
    BehaviorCLI.subcommand("microservice", app.ModelMicroserviceCLI)
    BehaviorCLI.run()
