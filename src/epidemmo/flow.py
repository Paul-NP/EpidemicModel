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

    def _get_factor_latex_repr(self, current_stage: Stage, start: Stage, content: str) -> str:
        if content == 'end':
            return f'v_{{{start.name[0].lower()}{current_stage.name[0].lower()}}}'
        else:
            return f'e_{{{current_stage.name[0].lower()},{self._short_name.lower()}}}'

    def _prepare_factors_dict(self, factors_data: dict[Stage, Factor | factorValue],
                              content: str, start: Stage) -> dict[Stage, Factor]:
        new_factors = {}
        for stage, factor in factors_data.items():
            if isinstance(factor, Factor):
                new_factors[stage] = factor
            else:
                factor_name = f'{content[:3]}[{stage.name[0]}]-{self}'
                factor = Factor(factor_name, factor)

                factor_repr = self._get_factor_latex_repr(stage, start, content)

                factor.set_latex_repr(factor_repr)
                new_factors[stage] = factor

        return new_factors

    def _prepare_flow_factor(self, factor_data: Factor | factorValue) -> Factor:
        if isinstance(factor_data, Factor):
            return factor_data
        else:
            factor = Factor(f'factor-{self}', factor_data)
            factor.set_latex_repr(f'v_{{{self._short_name}}}')
            return factor

    def __init__(self, start: Stage, end: stageFactorDict, flow_factor: anyFactor,
                 inducing: stageFactorDict) -> None:

        self._name, self._full_name = self._generate_names(start.name, [e.name for e in end.keys()],
                                                           [i.name for i in inducing.keys()])
        self._short_name = start.name + ''.join(e.name for e in end.keys())
        end_dict = self._prepare_factors_dict(end, 'end', start)
        flow_factor = self._prepare_flow_factor(flow_factor)
        ind_dict = self._prepare_factors_dict(inducing, 'inducing', start)

        self._population_size: float = 1.0
        self._relativity_factors: bool = False

        self._start: Stage = start
        self._end_dict: dict[Stage, Factor] = end_dict
        self._flow_factor: Factor = flow_factor

        self._ind_dict: dict[Stage, Factor] = ind_dict
        self._change_in: float = 0
        self._submit_func: Callable = self._teor_submit

    def set_population_size(self, population_size: float) -> None:
        self._population_size = population_size

    def set_relativity_factors(self, relativity: bool) -> None:
        self._relativity_factors = relativity

    def set_method(self, method: flowMethod) -> None:
        if method == self.TEOR_METHOD:
            self._submit_func = self._teor_submit
        elif method == self.STOCH_METHOD:
            self._submit_func = self._stoch_submit
        else:
            raise FlowError(f'flow have not calculation method = {method}')

    def _calc_flow_probability(self) -> None:
        if self._ind_dict:
            flow_factor = self._flow_factor.value
            if not self._relativity_factors:
                flow_factor /= self._population_size
            flow_probability = 1 - prod((1 - flow_factor * ind_factor.value) ** ind.num
                                        for ind, ind_factor in self._ind_dict.items())
        else:
            flow_probability = self._flow_factor.value
        self._flow_probability = flow_probability

    def calc_send_probability(self) -> None:
        self._calc_flow_probability()
        self._start.add_probability_out(self, self._flow_probability)

    def set_change_in(self, value: float) -> None:
        self._change_in = value

    def submit_changes(self) -> float:
        return self._submit_func()

    def check_end_factors(self) -> None:
        s = sum(f.value for f in self._end_dict.values())
        if abs(s - 1) > self._accuracy:
            raise FlowError(f'{self} sum of out probabilities not equal 1 ({s})')

    def _teor_submit(self) -> float:
        self._start.add_change(-self._change_in)
        for end, f in self._end_dict.items():
            end.add_change(f.value * self._change_in)
        return self._change_in

    def _stoch_submit(self) -> float:
        sum_ch = 0.0
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

    def send_latex_terms(self, simplified: bool) -> None:
        self._send_latex_out(simplified)
        self._send_latex_input(simplified)

    def _send_latex_out(self, simplified: bool) -> None:
        self._start.add_latex_out(self._get_latex_repr(simplified))

    def _send_latex_input(self, simplified: bool) -> None:
        full_repr = self._get_latex_repr(simplified)
        if len(self._end_dict.items()) > 1:
            for end, fa in self._end_dict.items():
                end.add_latex_input(f'{full_repr} \\cdot {fa.get_latex_repr()}')
        else:
            for end, fa in self._end_dict.items():
                end.add_latex_input(f'{full_repr}')

    def _get_latex_repr(self, simplified: bool) -> str:
        if self._ind_dict:
            inducing_part = self._get_inducing_part(simplified)
            if simplified:
                result = f'{self.start.get_latex_repr()} \\cdot {inducing_part}'
                if not self._relativity_factors:
                    return f'\\frac{{{result}}}{{N}}'
                return result

            return f'{self.start.get_latex_repr()} \\cdot {inducing_part}'
        else:
            return f'{self.start.get_latex_repr()} \\cdot {self._flow_factor.get_latex_repr()}'

    def _get_inducing_part(self, simplified: bool) -> str:
        full_factor = self._get_inducing_factor_part()
        simple_factor = self._flow_factor.get_latex_repr()

        if simplified:
            if len(self._ind_dict) == 1:
                st, fa = next(iter(self._ind_dict.items()))
                return f'{simple_factor} \\cdot {st.get_latex_repr()}'
            else:
                latex_sum = ' + '.join([f'{st.get_latex_repr()} \\cdot {fa.get_latex_repr()}'
                                        for st, fa in self._ind_dict.items()])
                return f'{simple_factor} \\cdot ({latex_sum})'
        else:
            if len(self._ind_dict) == 1:
                st, fa = next(iter(self._ind_dict.items()))
                return f'(1 - (1 - {full_factor})^{{{st.get_latex_repr()}}})'

            latex_prod = ' \\cdot '.join([f'(1 - {full_factor} \\cdot {fa.get_latex_repr()})^{{{st.get_latex_repr()}}}'
                                          for st, fa in self._ind_dict.items()])

            return f'(1 - {latex_prod})'

    def _get_inducing_factor_part(self):
        if self._relativity_factors:
            return self._flow_factor.get_latex_repr()

        return f'\\frac{{{self._flow_factor.get_latex_repr()}}}{{N}}'

    @staticmethod
    def _generate_names(start_name: str, end_names: list[str], ind_names: list[str]) -> tuple[str, str]:
        ends = ','.join(sorted(end_names))
        induced = ','.join(sorted(ind_names))
        return f'F({start_name}>{ends})', f'F({start_name}>{ends}|by-{induced})'

    def is_similar(self, other: Flow) -> bool:
        if self._start != other._start:
            return False
        if set(self._end_dict.keys()) & set(other._end_dict.keys()):
            return True
        return False

    def __str__(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return self._full_name

    @property
    def start(self) -> Stage:
        return self._start

    @property
    def ends(self) -> dict[Stage, Factor]:
        return self._end_dict

    @property
    def factor(self) -> Factor:
        return self._flow_factor

    @property
    def inducing(self) -> dict[Stage, Factor]:
        return self._ind_dict