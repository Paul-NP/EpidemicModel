from __future__ import annotations
from typing import Callable, TypeAlias, Optional, Union, Any

from .stage import Stage
from .factor import Factor

from scipy.stats import poisson  # type: ignore
from math import prod


factorValue: TypeAlias = Union[int, float, Callable[[int], float]]
anyFactor: TypeAlias = factorValue | Factor
stageFactorDict: TypeAlias = dict[Stage, anyFactor]
flowMethod: TypeAlias = int


class FlowError(Exception):
    pass


class Flow:
    _accuracy = 0.00001

    TEOR_METHOD: flowMethod = 0
    STOCH_METHOD: flowMethod = 1

    def _prepare_factors_dict(self, factors_data: dict[Stage, Factor | factorValue], content: str) -> dict[Stage, Factor]:
        new_factors = {}
        for stage, factor in factors_data.items():
            if isinstance(factor, Factor):
                new_factors[stage] = factor
            else:
                factor_name = f'{content[:3]}[{stage.name}]-{self}'
                new_factors[stage] = Factor(factor_name, factor)

        return new_factors

    def _prepare_flow_factor(self, factor_data: Factor | factorValue) -> Factor:
        if isinstance(factor_data, Factor):
            return factor_data
        else:
            return Factor(f'factor-{self}', factor_data)

    def __init__(self, start: Stage, end: stageFactorDict, flow_factor: anyFactor,
                 inducing: stageFactorDict):

        self._name = self._generate_flow_name(start.name, [e.name for e in end.keys()])
        end_dict = self._prepare_factors_dict(end, 'end')
        flow_factor = self._prepare_flow_factor(flow_factor)
        ind_dict = self._prepare_factors_dict(inducing, 'inducing')

        self._population_size: Optional[float | int] = None
        self._relativity_factors: bool = False

        self._start: Stage = start
        self._end_dict: dict[Stage, Factor] = end_dict
        self._flow_factor: Factor = flow_factor

        self._ind_dict: dict[Stage, Factor] = ind_dict
        self._change_in: float = 0
        self._submit_func: Callable = self._teor_submit

    def set_population_size(self, population_size: int | float):
        self._population_size = population_size

    def set_relativity_factors(self, relativity: bool):
        self._relativity_factors = relativity

    def set_method(self, method: flowMethod):
        if method == self.TEOR_METHOD:
            self._submit_func = self._teor_submit
        elif method == self.STOCH_METHOD:
            self._submit_func = self._stoch_submit
        else:
            raise FlowError(f'flow have not calculation method = {method}')

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

    @staticmethod
    def _generate_flow_name(start_name: str, end_names: list[str]):
        ends = ','.join(sorted(end_names))
        return f'F({start_name}>{ends})'

    def is_similar(self, other: Flow):
        if self._start != other._start:
            return False
        if set(self._end_dict.keys()) & set(other._end_dict.keys()):
            return True
        return False

    def __str__(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def start(self) -> Stage:
        return self._start

    @property
    def ends(self) -> list[Stage]:
        return list(self._end_dict.keys())
