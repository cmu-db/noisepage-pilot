import copy
import itertools

from action import ActionGenerator
from action import Action

from connector import Connector

from pglast import ast, stream
from pglast.enums.parsenodes import *

from enum import Enum


class KnobAction(Action):
    def __init__(self, name, setting: str = None, alterSystem = True):
        Action.__init__(self)
        self.name = name
        self.setting = setting
        self.alterSystem = alterSystem

    def _to_sql(self):
        setArg = None if self.setting is None else [ast.A_Const(ast.String(self.setting))]
        setKind = VariableSetKind.VAR_SET_DEFAULT if setArg is None else VariableSetKind.VAR_SET_VALUE
        self.ast = ast.VariableSetStmt(
            kind=setKind,
            name=self.name,
            args=setArg
        )
        sqlstr = stream.RawStream(semicolon_after_last_statement=True)(self.ast)
        return f'ALTER SYSTEM {sqlstr}' if self.alterSystem else sqlstr


class RelativeKnobType(Enum):
    DELTA = auto()
    PCT = auto()


class RelativeKnobGenerator(ActionGenerator):
    '''
    Create a ALTER SYSTEM stmt for a given knob name and numerical range
    '''

    def __init__(
        self,
        connector: Connector,
        name: str,
        type: RelativeKnobType = RelativeKnobType.PCT,
        minVal: float = 0.1,
        maxVal: float = 5,
        interval: float = 0.1
    ):
        ActionGenerator.__init__(self)
        self.connector = connector
        self.name = name
        self.type = type
        self.minVal = minVal
        self.maxVal = maxVal
        self.interval = interval

    def __iter__(self):
        val, unit = self.connector.get_config(self.name)
        print(val,unit)
        change = self.minVal
        while change <= self.maxVal:
            newVal = str(val)
            if self.type == RelativeKnobType.PCT:
                newVal = str(float(val) * change)
            elif self.type == RelativeKnobType.DELTA:
                newVal = str(float(val) + change)
            else:
                newVal = None
            yield KnobAction(self.name, newVal)
            change += self.interval
