from __future__ import annotations

from .fast_stage import FastStage
from .fast_factor import FastFactor
from .fast_flow import FastFlow
from .fast_model import FastEpidemicModel

from typing import Callable, TypeAlias, Union, Optional, Any

anyName: TypeAlias = str
stageName: TypeAlias = str
factorName: TypeAlias = str
factorValue: TypeAlias = Union[int, float, Callable[[int], float]]
stageNameFactorDict: TypeAlias = dict[stageName, factorName | factorValue]


class ModelBuilderError(Exception):
    pass


class FastModelBuilder:
    def __init__(self) -> None:
        self._stages: dict[str, FastStage] = {}
        self._factors: dict[str, FastFactor] = {}
        self._flows: list[FastFlow] = []

        self._relativity_factors: bool = False
        self._model_name: str = ''
        self._manual_name: bool = False

    def add_stage(self, name: stageName, start_num: int | float = 0, *,
                  latex_repr: Optional[str] = None) -> FastModelBuilder:
        self._check_name(name)
        self._check_new_stage_name(name)

        new_stage = FastStage(name, start_num, index=len(self._stages))
        new_stage.set_latext_repr(latex_repr)
        self._stages[name] = new_stage
        return self

    def add_stages(self, **kwargs: int | float) -> FastModelBuilder:
        for name, value in kwargs.items():
            self.add_stage(name, value)
        return self

    def add_factor(self, name: factorName, value: factorValue, *,
                   latex_repr: Optional[str] = None) -> FastModelBuilder:
        self._check_name(name)
        self._check_new_factor_name(name)

        new_factor = FastFactor(name, value)
        new_factor.set_latex_repr(latex_repr)
        self._factors[name] = new_factor
        return self

    def add_factors(self, **kwargs: factorValue) -> FastModelBuilder:
        for name, value in kwargs.items():
            self.add_factor(name, value)
        return self

    def add_flow(self, start: stageName, end: stageName | stageNameFactorDict,
                 factor: factorName | factorValue,
                 inducing: Optional[stageName | stageNameFactorDict] = None) -> FastModelBuilder:

        if inducing is None:
            inducing = {}

        self._check_start(start)
        self._check_stage_data(end, 'End')
        self._check_stage_data(inducing, 'Inducing')
        self._check_factor(factor)

        start_stage = self._prepare_start_stage(start)
        end_stages = self._prepare_stage_factor_dict(end)
        inducing_stages = self._prepare_stage_factor_dict(inducing)
        flow_factor = self._prepare_factor(factor)

        self._check_start_end_conflict(start_stage, end_stages)

        flow = FastFlow(start_stage, end_stages, flow_factor, inducing_stages, index=len(self._flows))
        self._check_new_flow(flow)

        self._flows.append(flow)
        return self

    def set_relativity_factors(self, relativity: bool) -> FastModelBuilder:
        if not isinstance(relativity, bool):
            raise ModelBuilderError('Relativity of Factors must be bool')
        self._relativity_factors = relativity
        return self

    def set_model_name(self, name: str) -> FastModelBuilder:
        self._check_name(name)
        self._manual_name = True
        self._model_name = name
        return self

    def build(self):
        self._check_completeness()
        if not self._manual_name:
            self._generate_model_name()

        stages = list(self._stages.values())
        flows = self._flows

        model = FastEpidemicModel(self._model_name, stages, flows, self._relativity_factors)
        model.run(1)

        return model

    def _check_start(self, start: Any):
        self._check_name(start)
        self._check_for_stage_name(start)

    @staticmethod
    def _check_start_end_conflict(start_stage: FastStage, end_stages: dict[FastStage, FastFactor | factorValue]):
        if start_stage in end_stages.keys():
            raise ModelBuilderError('Start Stage cannot coincide with any End Stage')

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
        if len(name) == 0:
            raise ModelBuilderError('Any name in the model cannot be empty')

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

    def _check_new_flow(self, flow: FastFlow):
        for fl in self._flows:
            if flow.is_similar(fl):
                raise ModelBuilderError(f'Two Flows connect the same Stages')

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

    def _check_factor(self, factor_data: Any):
        if isinstance(factor_data, factorName):
            self._check_for_factor_name(factor_data)
        elif not FastFactor.may_be_factor(factor_data):
            raise ModelBuilderError(f'Next value cannot be converted to a Factor: "{factor_data}"')

    def _prepare_start_stage(self, start_name: stageName) -> FastStage:
        return self._stages[start_name]

    def _prepare_factor(self, factor_data: factorName | factorValue) -> FastFactor | factorValue:
        if isinstance(factor_data, factorName):
            return self._factors[factor_data]
        else:
            return factor_data

    def _prepare_stage_factor_dict(self, source: stageName | stageNameFactorDict) -> dict[FastStage, FastFactor | factorValue]:
        if isinstance(source, stageName):
            stage_factor = self._get_one_stage_dict(source)
        else:
            stage_factor = self._get_many_stage_dict(source)

        return stage_factor

    def _get_one_stage_dict(self, source: stageName) -> dict[FastStage, FastFactor | factorValue]:
        return {self._stages[source]: 1}

    def _get_many_stage_dict(self, original_dict: stageNameFactorDict) -> dict[FastStage, FastFactor | factorValue]:
        stage_factor = {}
        for stage_name, factor in original_dict.items():
            stage = self._stages[stage_name]
            stage_factor[stage] = self._prepare_factor(factor)
        return stage_factor

    def _generate_model_name(self):
        self._model_name = ''.join([st_name[0].upper() for st_name in self._stages.keys()])

    def _check_completeness(self):
        start_stages = [fl.start for fl in self._flows]
        end_stages = [end for fl in self._flows for end in fl.ends.keys()]
        used_stages = set(start_stages + end_stages)

        not_used_stages = set(self._stages.values()) - used_stages
        if not_used_stages:
            raise ModelBuilderError(f'The following stages are not connected by Flows: {not_used_stages}')











