from __future__ import annotations
from typing import Sequence, Callable

from factor import Factor, FactorError
from stage import Stage, StageError
from flow import Flow, FlowError
from scipy.stats import poisson

import csv
import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable


class EpidemicModelError(Exception):
    pass


class EpidemicModel:
    __len_float: int = 4

    def __init__(self, stages: Sequence[Stage], flows: Sequence[Flow]):
        try:
            """
            EpidemicModel - compartmental epidemic model
            :param stages: list of Stages
            :param flows: list of Flows
            :param simulation_time: num of simulation steps
            :param imitation: flag for imitation or numeric simulation
            """
            if any(not isinstance(st, Stage) for st in stages):
                raise EpidemicModelError("all stages in model must be Stage")
            if any(not isinstance(fl, Flow) for fl in flows):
                raise EpidemicModelError("all flows in model must be Flow")

            self._stages: tuple[Stage] = tuple(stages)
            self._flows: tuple[Flow] = tuple(flows)
            self._factors: tuple[Factor] = tuple(set(fa for fl in self._flows for fa in fl.get_factors()))
            fa_names = tuple(fa.name for fa in self._factors)
            if len(set(fa_names)) < len(fa_names):
                raise EpidemicModelError('all factors must have different names')

            self._result_table: pd.DataFrame = pd.DataFrame(columns=[st.name for st in self._stages])
            self._factors_table: pd.DataFrame = pd.DataFrame(columns=[fa.name for fa in self._factors])
            self._flows_table: pd.DataFrame = pd.DataFrame(columns=[str(fl) for fl in self._flows])

        except Exception as e:
            raise type(e)(f'In init model: {e}')

    def _model_step(self, step: int):
        # print(f'step #{step} start')
        for fa in self._factors:
            fa.update(step)
            if fa.value < 0 or fa.value > 1:
                raise FactorError(f"'{fa.name}' value {fa.value} not in [0, 1]")

        for fl in self._flows:
            fl.check_end_factors()
            fl.calc_send_probability()

        for st in self._stages:
            st.send_out_flow()

        for fl in self._flows:
            fl.submit_changes()

        for st in self._stages:
            st.apply_changes()

        self._result_table.loc[step+1] = [st.num for st in self._stages]
        self._factors_table.loc[step] = [fa.value for fa in self._factors]
        self._flows_table.loc[step] = [fl.change for fl in self._flows]

    def start(self, time: int, stochastic_time=False, stochastic_changes=False, **kwargs):
        step = None
        try:
            old_factor_values = {}
            for f in self._factors:
                if f.name in kwargs:
                    old_factor_values[f] = f.get_fvalue()
                    f.set_fvalue(kwargs[f.name])

            self._result_table.drop(self._result_table.index, inplace=True)
            self._factors_table.drop(self._factors_table.index, inplace=True)
            self._flows_table.drop(self._factors_table.index, inplace=True)

            self._result_table.loc[0] = [st.num for st in self._stages]
            method = Flow.STOCH_METHOD if stochastic_changes else Flow.TEOR_METHOD
            for fl in self._flows:
                fl.set_method(method)

            if stochastic_time:
                step = poisson.rvs(mu=1)
                while step < time:
                    self._model_step(step)
                    step += poisson.rvs(mu=1)
            else:
                for step in range(0, time + 1):
                    self._model_step(step)

            for st in self._stages:
                st.reset_num()

            for fa in old_factor_values:
                fa.set_fvalue(old_factor_values[fa])

            return self.result_df

        except (FlowError, FactorError, StageError) as e:
            raise type(e)(f'in {'start' if step is None else 'step ' + step}: {e}')

    def _get_table(self, table_df: pd.DataFrame):
        table = PrettyTable()
        table.add_column('step', table_df.index.tolist())
        for col in table_df:
            table.add_column(col, table_df[col].tolist())
        table.float_format = f".{self.__len_float}"
        print(table)

    def print_result_table(self):
        print(self._get_table(self._result_table))

    def print_factors_table(self):
        print(self._get_table(self._factors_table))

    def print_flows_table(self):
        print(self._get_table(self._flows_table))

    def print_full_result(self):
        print(self._get_table(self.full_df))

    @property
    def result_df(self):
        return self._result_table.copy()

    @property
    def factors_df(self):
        return self._factors_table.copy()

    @property
    def flows_df(self):
        return self._flows_table.copy()

    @property
    def full_df(self):
        df = pd.concat([self._result_table, self._flows_table, self._factors_table], axis=1)
        df.sort_index(inplace=True)
        return df

    def _write_table(self, filename: str, table: pd.DataFrame, floating_point='.', delimiter=','):
        table.to_csv(filename, sep=delimiter, decimal=floating_point,
                     float_format=f'%.{self.__len_float}f', index_label='step')

    def write_all_result(self, model_name: str, floating_point='.', delimiter=',', path: str = ''):
        if path and path[-1] != '\\':
            path = path + '\\'
        if not isinstance(model_name, str) or len(model_name) == 0:
            raise EpidemicModelError('model name must be not empty str')
        self._write_table(f'{path}{model_name}_result.csv', self._result_table, floating_point, delimiter)
        self._write_table(f'{path}{model_name}_flows.csv', self._flows_table, floating_point, delimiter)
        self._write_table(f'{path}{model_name}_factors.csv', self._factors_table, floating_point, delimiter)


