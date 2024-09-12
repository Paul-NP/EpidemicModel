from __future__ import annotations
from typing import Sequence, Literal, Optional, Callable

from .factor import Factor, FactorError
from .stage import Stage, StageError
from .flow import Flow, FlowError, stageFactorDict

import pandas as pd  # type: ignore
import json
import numpy as np
from prettytable import PrettyTable
from scipy.stats import poisson  # type: ignore

from datetime import datetime


class EpidemicModelError(Exception):
    pass


class EpidemicModel:
    __len_float: int = 4

    __struct_versions = ['kk_2024']
    __struct_versions_types = Literal['kk_2024']

    def __init__(self, name: str, stages: list[Stage], flows: list[Flow], relativity_factors: bool) -> None:
        """
        EpidemicModel - compartmental epidemic model
        :param stages: list of Stages
        :param flows: list of Flows
        :param relativity_factors: if True, then the probability of flows will not be divided by the population
        size, otherwise it will be
        """
        try:
            self._name = name

            self._stages: tuple[Stage, ...] = tuple(stages)
            self._flows: tuple[Flow, ...] = tuple(flows)
            self._factors: tuple[Factor, ...] = tuple(set(fa for fl in self._flows for fa in fl.get_factors()))

            self._relativity_factors = False
            self.set_relativity_factors(relativity_factors)

            self._result_df: pd.DataFrame = pd.DataFrame(columns=[st.name for st in self._stages])
            self._factors_df: pd.DataFrame = pd.DataFrame(columns=[fa.name for fa in self._factors])
            self._flows_df: pd.DataFrame = pd.DataFrame(columns=[str(fl) for fl in self._flows])

        except Exception as e:
            raise type(e)(f'In init model: {e}')

    def _model_step(self, step: int) -> None:
        # print(f'step #{step} start')
        for fa in self._factors:
            fa.update(step)
            if fa.value < 0:
                raise FactorError(f"'{fa.name}' value {fa.value} not in [0, 1]")

        for fl in self._flows:
            fl.check_end_factors()
            fl.calc_send_probability()

        for st in self._stages:
            st.send_out_flow()

        fl_changes = []
        for fl in self._flows:
            fl_changes.append(fl.submit_changes())

        for st in self._stages:
            st.apply_changes()

        self._result_df.loc[step + 1] = [st.num for st in self._stages]
        self._factors_df.loc[step] = [fa.value for fa in self._factors]

        old_flows = self._flows_df.loc[step] if step in self._flows_df.index else 0
        self._flows_df.loc[step] = old_flows + np.array(fl_changes)

    def _set_population_size_flows(self, population_size: float) -> None:
        for flow in self._flows:
            flow.set_population_size(population_size)

    def set_relativity_factors(self, relativity: bool) -> None:
        if not isinstance(relativity, bool):
            raise EpidemicModelError('relativity_factors must be bool')
        for fl in self._flows:
            fl.set_relativity_factors(relativity)

    def start(self, time: int, stochastic_time=False, stochastic_changes=False, **kwargs) -> pd.DataFrame:
        step = -1
        try:
            population_size = sum(st.num for st in self._stages)
            self._set_population_size_flows(population_size)

            old_factor_values = self.set_factors(**kwargs)

            self._result_df.drop(self._result_df.index, inplace=True)
            self._factors_df.drop(self._factors_df.index, inplace=True)
            self._flows_df.drop(self._flows_df.index, inplace=True)

            self._result_df.loc[0] = [st.num for st in self._stages]
            method = Flow.STOCH_METHOD if stochastic_changes else Flow.TEOR_METHOD
            for fl in self._flows:
                fl.set_method(method)

            if stochastic_time:
                step = 0
                while step < time:
                    self._model_step(step)
                    step += poisson.rvs(mu=1)

            else:
                for step in range(0, time):
                    self._model_step(step)

            for st in self._stages:
                st.reset_num()

            for fa in old_factor_values:
                fa.set_fvalue(old_factor_values[fa])

            full_index = np.arange(time + 1)
            self._result_df = self._result_df.reindex(full_index, method='ffill')
            self._flows_df = self._flows_df.reindex(full_index, fill_value=0)
            self._factors_df = self._factors_df.reindex(full_index)
            return self.result_df

        except (FlowError, FactorError, StageError) as e:
            raise type(e)(f'in {"start" if step is None else "step " + str(step)}: {e}')

    def _get_table(self, table_df: pd.DataFrame) -> PrettyTable:
        table = PrettyTable()
        table.add_column('step', table_df.index.tolist())
        for col in table_df:
            table.add_column(col, table_df[col].tolist())
        table.float_format = f".{self.__len_float}"
        return table

    def print_result_table(self) -> None:
        print(self._get_table(self._result_df))

    def print_factors_table(self) -> None:
        print(self._get_table(self._factors_df))

    def print_flows_table(self) -> None:
        print(self._get_table(self._flows_df))

    def print_full_result(self) -> None:
        print(self._get_table(self.full_df))

    @property
    def name(self) -> str:
        return self._name

    @property
    def result_df(self) -> pd.DataFrame:
        return self._result_df.copy(deep=True)

    @property
    def factors_df(self) -> pd.DataFrame:
        return self._factors_df.copy(deep=True)

    @property
    def flows_df(self) -> pd.DataFrame:
        return self._flows_df.copy(deep=True)

    @property
    def full_df(self) -> pd.DataFrame:
        df = pd.concat([self._result_df, self._flows_df, self._factors_df], axis=1)
        df.sort_index(inplace=True)
        return df

    def _write_table(self, filename: str, table: pd.DataFrame, floating_point='.', delimiter=',') -> None:
        table.to_csv(filename, sep=delimiter, decimal=floating_point,
                     float_format=f'%.{self.__len_float}f', index_label='step')

    def write_results(self, floating_point='.', delimiter=',', path: str = '',
                      write_flows: bool = False, write_factors: bool = False) -> None:

        if path and path[-1] != '\\':
            path = path + '\\'

        current_time = datetime.today().strftime('%d_%b_%y_%H-%M-%S')
        first_part = f'{path}{self._name}_{current_time}'
        self._write_table(f'{first_part}_result.csv', self._result_df, floating_point, delimiter)
        if write_flows:
            self._write_table(f'{first_part}_flows.csv', self._flows_df, floating_point, delimiter)
        if write_factors:
            self._write_table(f'{first_part}_factors.csv', self._factors_df, floating_point, delimiter)

    def set_factors(self, **kwargs) -> dict[Factor, Callable[[int], float] | float]:
        old_factor_values = {}
        for f in self._factors:
            if f.name in kwargs:
                old_factor_values[f] = f.get_fvalue()
                f.set_fvalue(kwargs[f.name])

        return old_factor_values

    def set_start_stages(self, **kwargs) -> None:
        for s in self._stages:
            if s.name in kwargs:
                s.num = kwargs[s.name]

    def __str__(self) -> str:
        return f'Model({self._name})'

    def __repr__(self) -> str:
        return f'Model({self._name}): {list(self._flows)}'

    @property
    def stages(self) -> list[dict[str, float]]:
        return [{'name': st.name, 'num': st.start_num} for st in self._stages]

    @property
    def factors(self) -> list[dict[str, float]]:
        return [{'name': fa.name, 'value': 'dynamic' if fa.is_dynamic else fa.value} for fa in self._factors]

    @property
    def flows(self) -> list[dict]:
        flows = []
        for fl in self._flows:
            fl_dict = {'start': fl.start.name, 'factor': fl.factor.name,
                       'end': {st.name: fa.name for st, fa in fl.ends.items()},
                       'inducing': {st.name: fa.name for st, fa in fl.inducing.items()}}
            flows.append(fl_dict)
        return flows
