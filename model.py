from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from factor import Factor, FactorError
from stage import Stage
from flow import Flow


import csv
import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable


class EpidemicModelError(Exception):
    pass


class EpidemicModel:
    __len_float: int = 4

    def __init__(self, stages: list[Stage], flows: list[Flow], simulation_time: int,
                 imitation: bool = False):
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
            if not isinstance(simulation_time, int):
                raise EpidemicModelError("simulation_time must be Int")

            self._stages: list[Stage] = stages
            self._flows: list[Flow] = flows
            self._factors: list[Factor] = list(set(fa for fl in self._flows for fa in fl.get_factors()))
            self._simulation_time: int = simulation_time
            # self.imitation: bool = imitation
            self._result_table: pd.DataFrame = pd.DataFrame(columns=[st.name for st in self._stages])
            self._factors_table: pd.DataFrame = pd.DataFrame(columns=[fa.name for fa in self._factors])
            self._flows_table: pd.DataFrame = pd.DataFrame(columns=[str(fl) for fl in self._flows])
        except Exception as e:
            raise type(e)(f'In init model: {e}')

    # @property
    # def imitation(self):
    #     return self._imitation
    #
    # @imitation.setter
    # def imitation(self, value):
    #     if not isinstance(value, bool):
    #         raise ValueError("imitation for Flow must be bool")
    #     if self._imitation != value:
    #         for fl in self._flows:
    #             fl.imitation = value

    def _model_step(self, step: int):
        for fa in self._factors:
            fa.update(step)
            if fa.value < 0 or fa.value > 1:
                raise FactorError(f"'{fa.name}' value {fa.value} not in [0, 1]")

        for fl in self._flows:
            fl.calc_send_probability()

        for st in self._stages:
            st.send_out_flow()

        for fl in self._flows:
            fl.submit_changes()

        for st in self._stages:
            st.apply_changes()

        # self._result_table.append([step] + [st.num for st in self._stages])
        self._result_table.loc[step+1] = [st.num for st in self._stages]
        self._factors_table.loc[step] = [fa.value for fa in self._factors]
        self._flows_table.loc[step] = [fl.change for fl in self._flows]
        # print(f"#{step}: {self._model_result[-1]}")

    def start(self):
        step = None
        try:
            self._result_table.loc[0] = [st.num for st in self._stages]

            for step in range(0, self._simulation_time + 1):
                # print(f"step #{step}")
                self._model_step(step)
        except Exception as e:
            raise type(e)(f'in step {step}: {e}')

    def _print_table(self, table_df: pd.DataFrame):
        table = PrettyTable()
        table.add_column('step', table_df.index)
        for col in table_df:
            table.add_column(col, table_df[col])
        table.float_format = f".{self.__len_float}"
        print(table)

    def print_result_table(self):
        self._print_table(self._result_table)

    def print_factors_table(self):
        self._print_table(self._factors_table)

    def print_flows_table(self):
        self._print_table(self._flows_table)

    @property
    def result_df(self):
        return self._result_table.copy()

    @property
    def factors_df(self):
        return self._factors_table.copy()

    @property
    def flows_df(self):
        return self._flows_table.copy()

    def _write_table(self, filename: str, table: pd.DataFrame, floating_point='.', delimiter=','):
        table.to_csv(filename, sep=delimiter, decimal=floating_point,
                     float_format=f'%.{self.__len_float}f', index_label='step')

    def write_all_result(self, model_name: str, floating_point='.', delimiter=','):
        if not isinstance(model_name, str) or len(model_name) == 0:
            raise EpidemicModelError('model name must be not empty str')
        self._write_table(f'{model_name}_result.csv', self._result_table, floating_point, delimiter)
        self._write_table(f'{model_name}_flows.csv', self._flows_table, floating_point, delimiter)
        self._write_table(f'{model_name}_factors.csv', self._factors_table, floating_point, delimiter)


