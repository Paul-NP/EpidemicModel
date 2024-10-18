from __future__ import annotations
from typing import Callable, TypeAlias, Optional, Union, Any
from numpy.typing import NDArray

from .fast_stage import FastStage
from .fast_factor import FastFactor

from scipy.stats import poisson  # type: ignore
from math import prod


factorValue: TypeAlias = Union[int, float, Callable[[int], float]]
anyFactor: TypeAlias = factorValue | FastFactor
stageFactorDict: TypeAlias = dict[FastStage, anyFactor]
flowMethod: TypeAlias = int


class FastFlowError(Exception):
    pass


class FastFlow:
    _accuracy = 0.00001

    def _get_factor_latex_repr(self, current_stage: FastStage, start: FastStage, content: str) -> str:
        if content == 'end':
            return f'v_{{{start.name[0].lower()}{current_stage.name[0].lower()}}}'
        else:
            return f'e_{{{current_stage.name[0].lower()},{self._short_name.lower()}}}'

    def _prepare_factors_dict(self, factors_data: dict[FastStage, FastFactor | factorValue],
                              content: str, start: FastStage) -> dict[FastStage, FastFactor]:
        new_factors = {}
        for stage, factor in factors_data.items():
            if isinstance(factor, FastFactor):
                new_factors[stage] = factor
            else:
                factor_name = f'{content[:3]}[{stage.name[0]}]-{self}'
                factor = FastFactor(factor_name, factor)

                factor_repr = self._get_factor_latex_repr(stage, start, content)

                factor.set_latex_repr(factor_repr)
                new_factors[stage] = factor

        return new_factors

    def _prepare_flow_factor(self, factor_data: FastFactor | factorValue) -> FastFactor:
        if isinstance(factor_data, FastFactor):
            return factor_data
        else:
            factor = FastFactor(f'factor-{self}', factor_data)
            factor.set_latex_repr(f'v_{{{self._short_name}}}')
            return factor

    def __init__(self, start: FastStage, end: stageFactorDict, flow_factor: anyFactor,
                 inducing: stageFactorDict, *, index: int) -> None:

        self._name, self._full_name = self._generate_names(start.name, [e.name for e in end.keys()],
                                                           [i.name for i in inducing.keys()])
        self._index = index
        self._short_name = start.name + ''.join(e.name for e in end.keys())
        end_dict = self._prepare_factors_dict(end, 'end', start)
        flow_factor = self._prepare_flow_factor(flow_factor)
        ind_dict = self._prepare_factors_dict(inducing, 'inducing', start)

        self._relativity_factors: bool = False

        self._start: FastStage = start
        self._end_dict: dict[FastStage, FastFactor] = end_dict
        self._flow_factor: FastFactor = flow_factor
        self._ind_dict: dict[FastStage, FastFactor] = ind_dict

    @property
    def index(self) -> int:
        return self._index

    def connect_matrix(self, flows_weights: NDArray, targets: NDArray,
                       induction_weights: NDArray, outputs: NDArray) -> None:
        self._flow_factor.connect_matrix(flows_weights, self._index)

        for st, fa in self._ind_dict.items():
            fa.connect_matrix(induction_weights, (self._index, st.index))

        outputs[self._index, self._start.index] = 1

        for st, fa in self._end_dict.items():
            fa.connect_matrix(targets, (self._index, st.index))

    def get_factors(self) -> list[FastFactor]:
        all_factors = [self._flow_factor]
        for st, fa in self._ind_dict.items():
            all_factors.append(fa)
        for st, fa in self._end_dict.items():
            all_factors.append(fa)
        return all_factors

    def send_latex_terms(self, simplified: bool) -> None:
        self._send_latex_out(simplified)
        self._send_latex_input(simplified)

    def set_relativity_factors(self, relativity: bool) -> None:
        self._relativity_factors = relativity

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
    def start(self) -> FastStage:
        return self._start

    @property
    def ends(self) -> dict[FastStage, FastFactor]:
        return self._end_dict

    @property
    def factor(self) -> FastFactor:
        return self._flow_factor

    @property
    def inducing(self) -> dict[FastStage, FastFactor]:
        return self._ind_dict