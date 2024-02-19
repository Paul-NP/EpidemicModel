from __future__ import annotations
from typing import Callable
from factor import Factor, factor_one, factor_zero
from stage import Stage
from math import prod
from random import random


class Flow:
    _start: Stage
    _end: Stage
    _flow_factor: Factor
    _infect_factor: Factor
    _inducing_factors: dict[Stage, Factor]
    _imitation: bool

    _calc_changes_func: Callable[[Flow], float | int]

    __stage_name_len: int = 6
    __factor_float_len: int = 4

    def __init__(self, start: Stage, end: Stage, flow_factor: Factor = factor_one,
                 infect_factor: Factor = factor_one, inducing_factors: dict[Stage, Factor] = None,
                 imitation: bool = False):
        """
        Flow adds the changes it makes to the change lists of the corresponding Stages.
        The flow value is calculated based on parameter 'num' of the stages involved and
        parameter 'value' of the factors involved
        :param start: start Stage of Flow
        :param end: end Stage of Flow
        :param flow_factor: factor used in calculating the probability at the end or factor for not inducing Flow
        :param infect_factor: factor reflecting the probability of infection from one individual
        :param inducing_factors: dict of factors reflecting the influence of inducing stages
        """
        if not isinstance(start, Stage):
            raise ValueError("start of Flow must be Stage")
        if not isinstance(end, Stage):
            raise ValueError("end of Flow must be Stage")
        if start is end:
            raise ValueError("start Stage cannot coincide with end Stage")
        if not isinstance(flow_factor, Factor):
            raise ValueError("flow_factor must be Factor")
        if not isinstance(infect_factor, Factor):
            raise ValueError("infect_factor must be Factor")
        if inducing_factors is None:
            inducing_factors = {}
        if not isinstance(inducing_factors, dict):
            raise ValueError("inducing_factors must be dict")
        if any(not isinstance(key, Stage) or not isinstance(val, Factor) for key, val in inducing_factors.items()):
            raise ValueError("keys in inducing_factors must be Stage, values in inducing_factors must be Factor")

        self._start = start
        self._end = end
        self._flow_factor = flow_factor
        self._infect_factor = infect_factor
        self._inducing_factors = inducing_factors
        self.imitation = imitation

    @property
    def imitation(self):
        return self._imitation

    @imitation.setter
    def imitation(self, value):
        if not isinstance(value, bool):
            raise ValueError("imitation for Flow must be bool")
        if value:
            self._calc_changes_func = self._calc_imit_changes
        else:
            self._calc_changes_func = self._calc_numeric_changes

    def make_changes(self):
        # self._flow_factor(time)
        # self._infect_factor(time)
        # for f in self._inducing_factors.values():
        #     f(time)

        flow_value = self._calc_changes_func()
        self._start.add_change(-flow_value)
        self._end.add_change(flow_value)

    def _calc_imit_changes(self):
        # flow_value = sum(1 for _ in range(int(self._start.num)) if
        #                  (any(random() <= self._infect_factor.value * ind_factor.value
        #                       for ind, ind_factor in self._inducing_factors.items()
        #                       for _ in range(int(ind.num))) or not self._inducing_factors)
        #                  and random() < self._flow_factor.value)
        flow_probability = self._get_flow_probability()
        flow_value = sum(random() < flow_probability for _ in range(int(self._start.num)))
        return flow_value

    def _calc_numeric_changes(self):
        flow_probability = self._get_flow_probability()
        flow_value = self._start.num * flow_probability
        return flow_value

    def _get_flow_probability(self):
        if self._inducing_factors:
            not_infect_pr = prod((1 - self._infect_factor.value * ind_factor.value) ** ind.num
                                 for ind, ind_factor in self._inducing_factors.items())
        else:
            not_infect_pr = 0
        flow_probability = 1 - not_infect_pr
        flow_probability *= self._flow_factor.value
        return flow_probability

    def __str__(self) -> str:
        return f"Flow '{self._start.name}->{self._end.name}'"

    def __repr__(self) -> str:
        return f"Flow '{self._start.name:^{self.__stage_name_len}s}->{self._end.name:^{self.__stage_name_len}s}' | " \
               f"inf_f {self._infect_factor.value: .{self.__factor_float_len}f} | " \
               f"fl_f {self._flow_factor.value: .{self.__factor_float_len}f}"

    def __copy__(self) -> Flow:
        pass

    def copy(self) -> Flow:
        return self.__copy__()

    @property
    def flow_factor(self):
        return self._flow_factor

    @property
    def infect_factor(self):
        return self._infect_factor

    @property
    def inducing_factors(self):
        return self._inducing_factors

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
