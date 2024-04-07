from __future__ import annotations
from typing import Callable, TypeAlias, Optional
from types import FunctionType
from stage import Stage
from factor import Factor

from scipy.stats import poisson
from math import prod

from random import random


AnyFactor: TypeAlias = int | float | Callable[[int], float] | Factor


class FlowError(Exception):
    pass


class Flow:
    _accuracy = 0.00001

    TEOR_METHOD = 0
    STOCH_METHOD = 1

    def __init__(self, start: Stage, end: Stage | dict[Stage, AnyFactor], flow_factor: AnyFactor = 1,
                 inducing: Stage | dict[Stage, AnyFactor] = None):
        """
        Flow adds the changes it makes to the change lists of the corresponding Stages.
        The flow value is calculated based on parameter 'num' of the stages involved and
        parameter 'value' of the factors involved
        :param start: start Stage of Flow
        :param end: dict of factors
        :param flow_factor: the factor used in calculating the probability at the end or factor for not inducing Flow
        :param inducing: dict of factors reflecting the influence of inducing stages
        """
        if not isinstance(start, Stage):
            raise FlowError("start of Flow must be Stage")
        if isinstance(end, Stage):
            end = {end: Factor(1, name=None)}
        elif isinstance(end, dict):
            for k, v in end.items():
                if not isinstance(k, Stage):
                    raise FlowError("the end stages dictionary must include Stages as keys")
                elif Factor.may_be_factor(v):
                    end[k] = Factor(v, name=None)
                elif isinstance(v, Factor):
                    if v.name is None:
                        raise FlowError("Factors created manually must have names,"
                                        "one factor in end dict is unnamed")
                    end[k] = v
                else:
                    raise FlowError(f"the end stages dictionary must include {AnyFactor} as values")
        else:
            raise FlowError("end of Flow must be Stage or dict[Stage, <factor>]")
        if any(e is start for e in end):
            raise FlowError("start Stage cannot coincide with end Stage")
        if Factor.may_be_factor(flow_factor):
            flow_factor = Factor(flow_factor, name=None)
        elif not isinstance(flow_factor, Factor):
            raise FlowError(f"flow_factor must be {AnyFactor}")
        elif flow_factor.name is None:
            raise FlowError("Factors created manually must have names, flow_factor is unnamed")

        self._start: Stage = start
        self._end_dict: dict[Stage, Factor] = end
        self._flow_factor: Factor = flow_factor

        if inducing is None:
            inducing = {}
        elif isinstance(inducing, Stage):
            inducing = {inducing: 1}
        elif not isinstance(inducing, dict):
            raise FlowError("inducing must be dict[Stage, <factor>] or Stage")

        for k, v in inducing.items():
            if not isinstance(k, Stage):
                raise FlowError("keys in inducing must be Stage")
            elif Factor.may_be_factor(v):
                inducing[k] = Factor(v, name=None)
            elif not isinstance(v, Factor):
                raise FlowError(f"the inducing factors dictionary must include {AnyFactor} as values")
            elif v.name is None:
                raise FlowError("Factors created manually must have names, one of inducing factors is unnamed")

        self._inducing_factors: dict[Stage, Factor] = inducing
        self._change_in: float = 0
        self._submit_func: Callable = self._teor_submit

        self._rename_factors()

    def set_method(self, method: int):
        if method == self.TEOR_METHOD:
            self._submit_func = self._teor_submit
        elif method == self.STOCH_METHOD:
            self._submit_func = self._stoch_submit
        else:
            raise FlowError(f'flow have not calculation method = {method}')

    def _rename_factors(self):
        if self._flow_factor.name is None:
            self._flow_factor.name = f'{self}-f'
        for s, f in self._inducing_factors.items():
            if f.name is None:
                f.name = f'{self}-if[{s.name}]'
        for s, f in self._end_dict.items():
            if f.name is None:
                f.name = f'{self}-ef[{s.name}]'

    def _calc_flow_probability(self):
        if self._inducing_factors:
            not_infect_pr = prod((1 - self._flow_factor.value * ind_factor.value) ** ind.num
                                 for ind, ind_factor in self._inducing_factors.items())
            self._flow_probability = 1 - not_infect_pr
        else:
            self._flow_probability = self._flow_factor.value

    def calc_send_probability(self):
        self._calc_flow_probability()
        self._start.add_probability_out(self, self._flow_probability)

    def set_change_in(self, value: float):
        self._change_in = value

    def submit_changes(self):
        self._submit_func()

    def check_end_factors(self):
        s = sum(f.value for f in self._end_dict.values())
        if abs(s - 1) > self._accuracy:
            raise FlowError(f'{self} sum of out probabilities not equal 1 ({s})')

    def _teor_submit(self):
        self._start.add_change(-self._change_in)
        for end, f in self._end_dict.items():
            end.add_change(f.value * self._change_in)

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

    @property
    def change(self) -> float:
        return self._change_in

    def get_factors(self) -> list[Factor]:
        all_factors = [self._flow_factor]
        for st, fa in self._inducing_factors.items():
            all_factors.append(fa)
        for st, fa in self._end_dict.items():
            all_factors.append(fa)

        return all_factors

    def __str__(self) -> str:
        ends = ','.join([e.name for e in self._end_dict.keys()])
        return f"F({self._start.name}>{ends})"

    def __repr__(self) -> str:
        return self.__str__()
