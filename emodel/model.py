from __future__ import annotations
from typing import Sequence, Callable, Literal

import numpy as np

from factor import Factor, FactorError
from stage import Stage, StageError
from flow import Flow, FlowError
from scipy.stats import poisson

import csv
import pandas as pd
import json
import matplotlib.pyplot as plt
from prettytable import PrettyTable


class EpidemicModelError(Exception):
    pass


class EpidemicModel:
    __len_float: int = 4

    __struct_versions = ['kk_2024']
    __struct_versions_types = Literal['kk_2024']

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

            self._result_df: pd.DataFrame = pd.DataFrame(columns=[st.name for st in self._stages])
            self._factors_df: pd.DataFrame = pd.DataFrame(columns=[fa.name for fa in self._factors])
            self._flows_df: pd.DataFrame = pd.DataFrame(columns=[str(fl) for fl in self._flows])

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

        fl_changes = []
        for fl in self._flows:
            fl_changes.append(fl.submit_changes())

        for st in self._stages:
            st.apply_changes()

        self._result_df.loc[step + 1] = [st.num for st in self._stages]
        self._factors_df.loc[step] = [fa.value for fa in self._factors]

        old_flows = self._flows_df.loc[step] if step in self._flows_df.index else 0
        self._flows_df.loc[step] = old_flows + np.array(fl_changes)

    def start(self, time: int, stochastic_time=False, stochastic_changes=False, **kwargs):
        step = None
        try:
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
            raise type(e)(f'in {'start' if step is None else 'step ' + str(step)}: {e}')

    def _get_table(self, table_df: pd.DataFrame):
        table = PrettyTable()
        table.add_column('step', table_df.index.tolist())
        for col in table_df:
            table.add_column(col, table_df[col].tolist())
        table.float_format = f".{self.__len_float}"
        return table

    def print_result_table(self):
        print(self._get_table(self._result_df))

    def print_factors_table(self):
        print(self._get_table(self._factors_df))

    def print_flows_table(self):
        print(self._get_table(self._flows_df))

    def print_full_result(self):
        print(self._get_table(self.full_df))

    @property
    def result_df(self):
        return self._result_df.copy()

    @property
    def factors_df(self):
        return self._factors_df.copy()

    @property
    def flows_df(self):
        return self._flows_df.copy()

    @property
    def full_df(self):
        df = pd.concat([self._result_df, self._flows_df, self._factors_df], axis=1)
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
        self._write_table(f'{path}{model_name}_result.csv', self._result_df, floating_point, delimiter)
        self._write_table(f'{path}{model_name}_flows.csv', self._flows_df, floating_point, delimiter)
        self._write_table(f'{path}{model_name}_factors.csv', self._factors_df, floating_point, delimiter)

    def set_factors(self, **kwargs):
        old_factor_values = {}
        for f in self._factors:
            if f.name in kwargs:
                old_factor_values[f] = f.get_fvalue()
                f.set_fvalue(kwargs[f.name])

        return old_factor_values

    def set_start_stages(self, **kwargs):
        for s in self._stages:
            if s.name in kwargs:
                s.num = kwargs[s.name]

    def several_stoch_runs(self, n: int, time: int, stochastic_time=True, stochastic_changes=True, **kwargs):

        if not isinstance(n, int) or not isinstance(time, int):
            raise EpidemicModelError(f'several_stoch_runs expect two int but have {type(n)}, {type(time)}')

        # self.set_factors(**kwargs)
        results = {s.name: pd.DataFrame() for s in self._stages}
        flow_results = {str(fl): pd.DataFrame() for fl in self._flows}
        for i in range(1, n+1):
            print(f'stochastic run number {i:>4}')
            self.start(time, stochastic_time=stochastic_time, stochastic_changes=stochastic_changes, **kwargs)
            for st_name in self._result_df:
                results[st_name] = pd.concat([results[st_name], self._result_df[st_name]], axis=1)

            for fl_name in self._flows_df:
                flow_results[fl_name] = pd.concat([flow_results[fl_name], self._flows_df[fl_name]], axis=1)

        return results, flow_results

    def sir_with_data_delta(self, data):
        result_df = pd.DataFrame(columns=list('SIR'))
        result_df.loc[0] = [st.num for st in self._stages]

        gama = [fa for fa in self._factors if fa.name == 'gama'][0]
        for step in range(len(data)):
            si = data[step]
            ir = min(result_df.loc[step, 'I'], poisson.rvs(result_df.loc[step, 'I'] * gama.value))
            changes = np.array([-si, si - ir, ir])
            result_df.loc[step+1] = result_df.loc[step] + changes

        return result_df

    @staticmethod
    def get_sir(s=1000, i=1, r=0, beta=0.0004, gama=0.1):
        match s, i, r, beta, gama:
            case int(s), int(i), int(r), float(beta), float(gama):
                s = Stage('S', s)
                i = Stage('I', i)
                r = Stage('R', r)

                beta = Factor(beta, name='beta')
                gama = Factor(gama, name='gama')

                si = Flow(s, i, beta, inducing_factors=i)
                ir = Flow(i, r, gama)

                model = EpidemicModel((s, i, r), (si, ir))
                return model
            case _:
                raise EpidemicModelError('cannot create sir model with taken parameters')

    # def confidence_interval(self, results, p_value: float = 0.9):
    #     intervals = {}
    #     percent = int(p_value * 100)
    #     percent_ = 100 - percent
    #     for r in results:
    #         low = np.percentile(results[r], percent_, axis=1)
    #         upp = np.percentile(results[r], percent, axis=1)
    #         intervals[r] = pd.DataFrame({'l': low, 'u': upp}, index=results[r].index)
    #     return intervals

    def to_json(self, struct_version: __struct_versions_types) -> str:
        generators = {'kk_2024': self.__to_json_kk_2024}
        if struct_version in generators:
            return generators[struct_version]()
        else:
            raise EpidemicModelError('unknown json-structure version')

    @classmethod
    def from_json(cls, json_string: str, struct_version: __struct_versions_types) -> EpidemicModel:
        parsers = {'kk_2024': cls.__from_json_kk_2024}

        if struct_version in parsers:
            return parsers[struct_version](json_string)
        else:
            raise EpidemicModelError('unknown json-structure version')

    def __to_json_kk_2024(self) -> str:
        pass

    @classmethod
    def __from_json_kk_2024(cls, json_string: str) -> EpidemicModel:
        structure = json.loads(json_string)
        try:
            raw_stages = structure['compartments']
            raw_flows = structure['flows']

            stages = [Stage(st['name'], st['population']) for st in raw_stages]
            stages_dict = {st.name: st for st in stages}

            flows = []
            for r_flow in raw_flows:
                start = stages_dict[r_flow['from']]

                end_dict = {stages_dict[end['name']]: end['coef'] for end in r_flow['to']}
                if 'induction' in r_flow:
                    ind_dict = {stages_dict[ind['name']]: ind['coef'] for ind in r_flow['induction']}
                else:
                    ind_dict = None
                fl_factor = r_flow['coef']
                flows.append(Flow(start, end_dict, fl_factor, ind_dict))

            model = EpidemicModel(stages, flows)
            return model

        except Exception as e:
            e.add_note('incorrect json structure in version "kk_2024"')
            raise e






