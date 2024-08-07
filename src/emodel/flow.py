from __future__ import annotations
from typing import Callable, TypeAlias, Optional, Union

from .stage import Stage
from .factor import Factor

from scipy.stats import poisson  # type: ignore
from math import prod


AnySourceFactor: TypeAlias = Union[int, float, Callable[[int], float]]
AnyFactor: TypeAlias = AnySourceFactor | Factor
StageFactorDict: TypeAlias = dict[Stage, AnyFactor]
FlowMethod: TypeAlias = int


class FlowError(Exception):
    pass


class Flow:
    _accuracy = 0.00001

    TEOR_METHOD: FlowMethod = 0
    STOCH_METHOD: FlowMethod = 1

    @staticmethod
    def __check_factors_dict(factors: StageFactorDict, content: str) -> dict[Stage, Factor]:
        if not factors:
            raise FlowError(f'{content} dictionary is empty')
        new_factors = {}
        for k, v in factors.items():
            if not isinstance(k, Stage):
                raise FlowError(f"{content} dictionary must include Stages as keys")
            elif isinstance(v, Factor):
                if not v.name:
                    raise FlowError(f"Factors created manually must have names,"
                                    f"one factor in {content} is unnamed")

                new_factors[k] = v
            elif Factor.may_be_factor(v):
                new_factors[k] = Factor(v, name=None)
            else:
                raise FlowError(f"{content} dictionary must include {AnyFactor} as values")
        return new_factors

    def __init__(self, start: Stage, end: Stage | StageFactorDict,
                 flow_factor: AnyFactor = 1, inducing_factors: Optional[Stage | StageFactorDict] = None):

        if not isinstance(start, Stage):
            raise FlowError("start of Flow must be Stage")

        if isinstance(end, Stage):
            end_dict = {end: Factor(1, name=None)}
        elif isinstance(end, dict):
            end_dict = self.__check_factors_dict(end, 'end')
        else:
            raise FlowError(f"end of Flow must be Stage or dict[Stage, {AnyFactor}]")

        if any(e is start for e in end_dict):
            raise FlowError("start Stage cannot coincide with end Stage")

        if isinstance(flow_factor, Factor):
            if not flow_factor.name:
                raise FlowError(f"Factors created manually must have names,"
                                f"flow_factor is unnamed")
        elif Factor.may_be_factor(flow_factor):
            flow_factor = Factor(flow_factor, name=None)
        else:
            raise FlowError(f'flow_factor must be {AnyFactor}')

        if isinstance(inducing_factors, Stage):
            inducing_dict = {inducing_factors: Factor(1, name=None)}
        elif isinstance(inducing_factors, dict):
            inducing_dict = self.__check_factors_dict(inducing_factors, 'factors')
        elif inducing_factors is None:
            inducing_dict = {}
        else:
            raise FlowError(f'inducing must be {Stage} or dict[Stage, {AnyFactor}]')

        self._population_size: Optional[float | int] = None
        self._relativity_factors: bool = False

        self._start: Stage = start
        self._end_dict: dict[Stage, Factor] = end_dict
        self._flow_factor: Factor = flow_factor

        self._ind_dict: dict[Stage, Factor] = inducing_dict
        self._change_in: float = 0
        self._submit_func: Callable = self._teor_submit

        self._rename_factors()

    def set_population_size(self, population_size: int | float):
        self._population_size = population_size

    def set_relativity_factors(self, relativity: bool):
        self._relativity_factors = relativity

    def set_method(self, method: FlowMethod):
        if method == self.TEOR_METHOD:
            self._submit_func = self._teor_submit
        elif method == self.STOCH_METHOD:
            self._submit_func = self._stoch_submit
        else:
            raise FlowError(f'flow have not calculation method = {method}')

    def _rename_factors(self):
        if self._flow_factor is not None and not self._flow_factor.name:
            self._flow_factor.name = f'{self}-f'
        for s, f in self._ind_dict.items():
            if not f.name:
                f.name = f'if[{s.name}]-{self}'
        for s, f in self._end_dict.items():
            if not f.name:
                f.name = f'ef[{s.name}]-{self}'

    def _calc_flow_probability(self):
        if self._ind_dict:
            flow_factor = self._flow_factor.value
            if not self._relativity_factors:
                flow_factor /= self._population_size
            flow_probability = 1 - prod((1 - flow_factor * ind_factor.value) ** ind.num
                                        for ind, ind_factor in self._ind_dict.items())
        else:
            flow_probability = self._flow_factor.value
        self._flow_probability = flow_probability

    def calc_send_probability(self):
        self._calc_flow_probability()
        self._start.add_probability_out(self, self._flow_probability)

    def set_change_in(self, value: float):
        self._change_in = value

    def submit_changes(self):
        return self._submit_func()

    def check_end_factors(self):
        s = sum(f.value for f in self._end_dict.values())
        if abs(s - 1) > self._accuracy:
            raise FlowError(f'{self} sum of out probabilities not equal 1 ({s})')

    def _teor_submit(self):
        self._start.add_change(-self._change_in)
        for end, f in self._end_dict.items():
            end.add_change(f.value * self._change_in)
        return self._change_in

    def _stoch_submit(self):
        sum_ch = 0
        for end, f in self._end_dict.items():
            ch = poisson.rvs(mu=f.value * self._change_in)
            sum_ch_new = sum_ch + ch
            if sum_ch_new > self._start.num:
                end.add_change(self._start.num - sum_ch)
                sum_ch = self._start.num
                break
            else:
                sum_ch = sum_ch_new
                end.add_change(ch)

        self._start.add_change(-sum_ch)
        return sum_ch

    @property
    def change(self) -> float:
        return self._change_in

    def get_factors(self) -> list[Factor]:
        all_factors = [self._flow_factor]
        for st, fa in self._ind_dict.items():
            all_factors.append(fa)
        for st, fa in self._end_dict.items():
            all_factors.append(fa)

        return all_factors

    def __str__(self) -> str:
        ends = ','.join([e.name for e in self._end_dict.keys()])
        return f"F({self._start.name}>{ends})"

    def __repr__(self) -> str:
        return self.__str__()
