import logging

from plumbum import cli

from behavior.datagen import gen
from behavior.modeling import train
from behavior.plans import diff

logger = logging.getLogger(__name__)


class BehaviorCLI(cli.Application):
    def main(self) -> None:
        pass


if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s:%(asctime)s %(message)s", level=logging.INFO
    )
    BehaviorCLI.subcommand("datagen", gen.DataGeneratorCLI)
    BehaviorCLI.subcommand("datadiff", diff.DataDiffCLI)
    BehaviorCLI.subcommand("train", train.TrainCLI)
    BehaviorCLI.run()
