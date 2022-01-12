#!/usr/bin/env python3

from plumbum import FG
from plumbum.cmd import python3


training_data_folder = "./artifacts/behavior/datagen/training_data"
diff_data_folder = "./artifacts/behavior/datagen/differenced_data"
datadiff_args = ["--dir-datagen-data", training_data_folder, "--dir-output", diff_data_folder]
python3["-m", "behavior", "datadiff", datadiff_args] & FG
