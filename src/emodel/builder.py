from .stage import Stage
from .factor import Factor
from .flow import Flow
from .model import EpidemicModel

from typing import Callable, TypeAlias, Union, Optional, Any


anyName: TypeAlias = str
stageName: TypeAlias = str
factorName: TypeAlias = str
factorValue: TypeAlias = Union[int, float, Callable[[int], float]]
stageNameFactorDict: TypeAlias = dict[stageName, factorName | factorValue]


class ModelBuilderError(Exception):
    pass


class ModelBuilder:
    def __init__(self):
        self._stages = {}
        self._factors = {}
        self._flows = []

    def add_stage(self, name: stageName, start_num: int | float) -> None:
        self._check_name(name)
        self._check_new_stage_name(name)

        new_stage = Stage(name, start_num)
        self._stages[name] = new_stage

    def add_factor(self, name: factorName, value: factorValue) -> None:
        self._check_name(name)
        self._check_new_factor_name(name)

        new_factor = Factor(name, value)
        self._factors[name] = new_factor

    def add_flow(self, start: stageName, end: stageName | stageNameFactorDict,
                 factor: factorName | factorValue,
                 inducing: Optional[stageName | stageNameFactorDict] = None) -> None:

        if inducing is None:
            inducing = {}

        self._check_start(start)
        self._check_stage_data(end, 'End')
        self._check_stage_data(inducing, 'Inducing')

        flow_name = Flow.generate_flow_name(start)

        start_stage = self._prepare_start_stage(start)
        end_stages = self._get_stage_factor_dict(end)
        inducing_stages = self._get_stage_factor_dict(inducing)

        flow_factor = self._get_factor(factor)

        flow = Flow(start_stage, end_stages, flow_factor, inducing_stages)
        self._flows.append(flow)

    def _check_start(self, start: Any):
        self._check_name(start)
        self._check_for_stage_name(start)

    def _check_stage_data(self, data: Any, source_type: str):
        if isinstance(data, stageName):
            self._check_for_stage_name(data)
        elif isinstance(data, dict):
            self._check_names_dict(data)
            self._check_dict_for_stage_name(data)
            self._check_dict_for_factor_name(data)
        else:
            raise ModelBuilderError(f'The {source_type} for a Flow should be a Stage name or '
                                    f'a dictionary of Stages and Factors')

    @staticmethod
    def _check_name(name: Any):
        if not isinstance(name, anyName):
            raise ModelBuilderError('Any name in the model must be a string')

    @staticmethod
    def _check_names_dict(original_dict: dict[Any, Any]):
        if any(not isinstance(name, anyName) for name in original_dict.keys()):
            raise ModelBuilderError('Any name in the model must be a string')

    def _check_new_stage_name(self, name: stageName):
        if name in self._stages:
            raise ModelBuilderError(f'Stage named "{name}" has already been added')

    def _check_new_factor_name(self, name: factorName):
        if name in self._factors:
            raise ModelBuilderError(f'Factor named "{name}" has already been added')

    def _check_for_stage_name(self, name: stageName):
        if name not in self._stages:
            raise ModelBuilderError(f'Stage "{name}" is not defined')

    def _check_dict_for_stage_name(self, name_dict: dict[stageName, Any]):
        potential_names = set(name_dict.keys())
        existing_names = set(self._stages.keys())
        new_names = potential_names - existing_names
        if new_names:
            raise ModelBuilderError(f'Stages with names: {new_names} are not defined')

    def _check_for_factor_name(self, name: factorName):
        if name not in self._factors:
            raise ModelBuilderError(f'Factor "{name}" is not defined')

    def _check_dict_for_factor_name(self, factor_dict: dict[stageName, Any]):
        potential_names = set(factor for factor in factor_dict.values() if isinstance(factor, factorName))
        existing_names = set(self._factors.keys())
        new_names = potential_names - existing_names
        if new_names:
            raise ModelBuilderError(f'Factors with names: {new_names} are not defined')

    def _prepare_start_stage(self, start_name: stageName) -> Stage:
        return self._stages[start_name]

    def _get_factor(self, original_factor: factorName | factorValue) -> Factor | factorValue:
        if isinstance(original_factor, factorName):
            self._check_for_factor_name(original_factor)
            return self._factors[original_factor]
        else:
            return original_factor

    def _get_stage_factor_dict(self, source: stageName | stageNameFactorDict) -> dict[Stage, Any]:
        if isinstance(source, stageName):
            stage_factor = self._get_one_stage_dict(source)
        else:
            stage_factor = self._get_many_stage_dict(source)

        return stage_factor

    def _get_one_stage_dict(self, source: stageName) -> dict[Stage, int]:
        return {self._stages[source]: 1}

    def _get_many_stage_dict(self, original_dict: stageNameFactorDict) -> dict[Stage, Any]:
        stage_factor = {}
        for stage_name, factor in original_dict.items():
            stage = self._stages[stage_name]
            stage_factor[stage] = self._get_factor(factor)
        return stage_factor

            






