from .stage import Stage
from .factor import Factor
from .flow import Flow
from .model import EpidemicModel

from typing import Callable, TypeAlias, Union, Optional

stageName: TypeAlias = str
factorName: TypeAlias = str
factorValue: TypeAlias = Union[int, float, Callable[[int], float]]
stageFactorDict: TypeAlias = dict[stageName, factorName | factorValue]


class ModelBuilderError(Exception):
    pass


class ModelBuilder:
    def __init__(self):
        self._stages = {}
        self._factors = {}
        self._flows = []

    def add_stage(self, name: stageName, start_num: int | float) -> None:
        new_stage = Stage(name, start_num)
        if name in self._stages:
            raise ModelBuilderError(f'Stage named "{name}" has already been added')
        self._stages[name] = new_stage

    def add_factor(self, name: factorName, value: factorValue) -> None:
        new_factor = Factor(name, value)
        if name in self._factors:
            raise ModelBuilderError(f'Factor named "{name}" has already been added')
        self._factors[name] = new_factor

    def add_flow(self, start_name: stageName, end_names: stageName | stageFactorDict,
                 flow_factor: factorName | factorValue,
                 inducing_factors: Optional[stageName | stageFactorDict] = None) -> None:
        if start_name not in self._stages:
            raise ModelBuilderError('Flow begins at an unknown stage')
        if isinstance(end_names, stageName) and end_names not in self._stages:
            raise ModelBuilderError('Flow ends at an unknown stage')
        if isinstance(end_names, dict):
            if not set(end_names.keys()).issubset(self._stages.keys()):
                raise ModelBuilderError('Flow ends at an unknown stage')
            for end_factor in end_names.values():
                if isinstance(end_factor, str) and end_factor not in self._factors:
                    raise ModelBuilderError('Flow ends with unknown factor usage')

        if inducing_factors is not None and not set(inducing_factors.keys()).issubset(self._stages.keys()):
            raise ModelBuilderError('Flow induced by unknown stage')







