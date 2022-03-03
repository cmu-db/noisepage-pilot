# doit automatically picks up tasks as long as their unqualified name is prefixed with task_.
# Read the guide: https://pydoit.org/tasks.html

from dodos.action import *
from dodos.behavior import *
from dodos.benchbase import *
from dodos.ci import *
from dodos.forecast import *
from dodos.noisepage import *
from dodos.pilot import *
from dodos.project1 import *
from dodos.tscout import *

DOIT_CONFIG = {
    'backend': "sqlite3"
}
