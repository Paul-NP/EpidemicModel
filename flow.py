from __future__ import annotations
from typing import Callable, TypeAlias, Optional
from types import FunctionType
from stage import Stage
from factor import Factor

from math import prod

from random import random


AnyFactor: TypeAlias = int | float | Callable[[int], float] | Factor

class FlowError(Exception):
    pass


class Flow:
    _accuracy = 0.00001

    def __init__(self, start: Stage, end: Stage | dict[Stage, AnyFactor], flow_factor: AnyFactor = 1,
                 inducing: dict[Stage, AnyFactor] = None):
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
            end = {end: Factor(1)}
        elif isinstance(end, dict):
            for k, v in end.items():
                if not isinstance(k, Stage):
                    raise FlowError("the end stages dictionary must include Stages as keys")
                elif Factor.may_be_factor(v):
                    end[k] = Factor(v)
                elif isinstance(v, Factor):
                    end[k] = v
                else:
                    raise FlowError(f"the end stages dictionary must include {AnyFactor} as values")
        else:
            raise FlowError("end of Flow must be Stage or dict[Stage, <factor>]")
        if any(e is start for e in end):
            raise FlowError("start Stage cannot coincide with end Stage")
        if Factor.may_be_factor(flow_factor):
            flow_factor = Factor(flow_factor)
        elif not isinstance(flow_factor, Factor):
            raise FlowError(f"flow_factor must be {AnyFactor}")

        self._start: Stage = start
        self._end_dict: dict[Stage, Factor] = end
        self._flow_factor: Factor = flow_factor

        if inducing is None:
            inducing = {}
        if not isinstance(inducing, dict):
            raise FlowError("inducing_factors must be dict")

        for k, v in inducing.items():
            if not isinstance(k, Stage):
                raise FlowError("keys in inducing_factors must be Stage")
            elif Factor.may_be_factor(v):
                inducing[k] = Factor(v)
            elif not isinstance(v, Factor):
                raise FlowError(f"the inducing factors dictionary must include {AnyFactor} as values")

        self._inducing_factors: dict[Stage, Factor] = inducing
        self._change_in: float = 0

        self._rename_factors()

    def _rename_factors(self):
        if self._flow_factor.name is None:
            self._flow_factor.name = f'{self} - flow factor'
        for s, f in self._inducing_factors.items():
            if f.name is None:
                f.name = f'{self} - ind factor by {s}'
        for s, f in self._end_dict.items():
            if f.name is None:
                f.name = f'{self} - end factor to {s}'

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
        self._start.add_change(-self._change_in)
        s = 0
        for end, f in self._end_dict.items():
            s += f.value
            end.add_change(f.value * self._change_in)
        if abs(s - 1) > self._accuracy:
            raise FlowError(f'{self} sum of out probabilities not equal 1 ({s})')

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
        return f"Flow({self._start.name}>{ends})"

    def __repr__(self) -> str:
        return self.__str__()


# st1 = Stage("S", 100)
# st2 = Stage("I", 1)
# print(repr(st1), repr(st2))
#
# beta = Factor(0.43, "beta")
# gama = Factor(0.1, "gama")
#
# fl = Flow(st1, st2, infect_factor=beta, inducing_factors={st2: gama})
# fl.imitation = True
# print(repr(fl))
# beta(1)
# gama(1)
#
# fl.make_changes()
# st1.apply_changes()
# st2.apply_changes()
# print(repr(st1), repr(st2))
