from datetime import datetime

import numpy as np
from itertools import product
import pandas as pd
from prettytable import PrettyTable

from .flow import Flow
from .stage import Stage
from .factor import Factor
from scipy.stats import poisson


from typing import Optional, Callable


class ModelError(Exception):
    pass


class EpidemicModel:
    __len_float: int = 4

    def __init__(self, name: str, stages: list[Stage], flows: list[Flow], relativity_factors: bool):
        self._name: str = name

        stages = sorted(stages, key=lambda st: st.index)  # сортируем стадии по индексу
        flows = sorted(flows, key=lambda fl: fl.index)  # сортируем потоки по индексу

        self._stages: tuple[Stage, ...] = tuple(stages)
        self._flows: tuple[Flow, ...] = tuple(flows)
        self._factors: tuple[Factor, ...] = tuple(set(fa for fl in flows for fa in fl.get_factors()))

        self._stage_names: tuple[str, ...] = tuple(st.name for st in stages)
        self._flow_names: tuple[str, ...] = tuple(str(fl) for fl in flows)
        self._factors_names: tuple[str, ...] = tuple(fa.name for fa in self._factors)

        self._stage_starts: np.ndarray = np.array([st.start_num for st in stages], dtype=np.float64)

        # факторы, которые будут изменяться во время моделирования
        self._dynamic_factors: list[Factor] = [fa for fa in self._factors if fa.is_dynamic]

        self._flows_weights: np.ndarray = np.zeros(len(flows), dtype=np.float64)
        self._targets: np.ndarray = np.zeros((len(flows), len(stages)), dtype=np.float64)
        self._induction_weights: np.ndarray = np.zeros((len(flows), len(stages)), dtype=np.float64)
        self._outputs: np.ndarray = np.zeros((len(flows), len(stages)), dtype=np.bool)

        # связываем факторы, используемые в потоках, с матрицами
        self._connect_matrix(flows)

        self._duration = 1
        self._result: np.ndarray = np.zeros((self._duration, len(stages)), dtype=np.float64)
        self._result[0] = self._stage_starts
        self._result_flows: Optional[np.ndarray] = None
        self._result_factors: Optional[np.ndarray] = None

        self._flows_probabs: np.ndarray = np.zeros(len(flows), dtype=np.float64)
        self._flows_values: np.ndarray = np.zeros(len(flows), dtype=np.float64)
        self._induction_mask: np.ndarray = self._induction_weights.any(axis=1)
        self._induction: np.ndarray = self._induction_weights[self._induction_mask]

        self._relativity_factors: bool = False
        self.set_relativity_factors(relativity_factors)

    def set_relativity_factors(self, relativity_factors: bool):
        if not isinstance(relativity_factors, bool):
            raise ModelError('relativity_factors must be bool')
        for fl in self._flows:
            fl.set_relativity_factors(relativity_factors)
        self._relativity_factors = relativity_factors

    def _update_all_factors(self):
        for fa in self._factors:
            fa.update(0)

    def _update_dynamic_factors(self, step: int):
        for fa in self._dynamic_factors:
            fa.update(step-1)

    def _prepare_factors_matrix(self, *args):
        self._iflow_weights = self._flows_weights[self._induction_mask].reshape(-1, 1)
        self._flows_probabs[~self._induction_mask] = self._flows_weights[~self._induction_mask]
        self._check_matrix()

    def _check_matrix(self):
        if (self._targets.sum(axis=1) != 1).any():
            raise ModelError('Sum of targets one Flow must be 1')
        if (self._flows_weights < 0).any():
            raise ModelError('Flow weights must be >= 0')
        if (self._flows_weights[~self._induction_mask] > 1).any():
            raise ModelError('Not Induction Flow weights must be in range [0, 1]')
        if self._relativity_factors and (self._flows_weights[self._induction_mask] > 1).any():
            raise ModelError('Induction Flow weights, if they are relativity, must be in range [0, 1]')
        if ((self._induction_weights < 0) | (self._induction_weights > 1)).any():
            raise ModelError('Induction weights must be in range [0, 1]')

    def _correct_not_rel_factors(self, *args):
        self._flows_weights[self._induction_mask] /= self._stage_starts.sum()

    def _connect_matrix(self, flows: list[Flow]):
        for fl in flows:
            fl.connect_matrix(self._flows_weights, self._targets, self._induction_weights, self._outputs)

    def _prepare(self):
        self._update_all_factors()
        if not self._relativity_factors:
            self._correct_not_rel_factors()
        self._prepare_factors_matrix()

    def start(self, duration: int, *, full_save: bool = False, stochastic: bool = False) -> pd.DataFrame:
        self._duration = duration
        self._start(full_save, stochastic)

        df = self._get_result_df()
        return df

    def _start(self, save_full: bool, stochastic: bool):
        self._result = np.zeros((self._duration, len(self._stage_starts)), dtype=np.float64)
        self._result[0] = self._stage_starts

        self._prepare()

        if not self._dynamic_factors and not save_full and not stochastic:
            self._fast_run()
            return

        self._full_step_seq: list[Callable] = []
        if self._dynamic_factors:
            self._full_step_seq.append(self._update_dynamic_factors)
            if not self._relativity_factors:
                self._full_step_seq.append(self._correct_not_rel_factors)
            self._full_step_seq.append(self._prepare_factors_matrix)

        if stochastic:
            self._full_step_seq.append(self._stoch_step)
        else:
            self._full_step_seq.append(self._determ_step)

        if save_full:
            self._full_step_seq.append(self._save_additional_results)
            self._result_flows = np.zeros((self._duration, len(self._flow_names)), dtype=np.float64)
            self._result_factors = np.zeros((self._duration, len(self._factors_names)), dtype=np.float64)
        else:
            self._result_flows = None
            self._result_factors = None

        if stochastic:
            self._stoch_run()
        else:
            self._determ_run()

    def _determ_run(self):
        for step in range(1, self._duration):
            for step_func in self._full_step_seq:
                step_func(step)

    def _stoch_run(self):
        step = 1
        while step < self._duration:
            for step_func in self._full_step_seq:
                step_func(step)
            new_step = step + poisson.rvs(mu=1)
            self._result[step: new_step] = self._result[step]
            step = new_step

    def _fast_run(self):
        for step in range(1, self._duration):
            pr = self._result[step - 1]
            a = np.array([1,2,3])
            self._induction * self._iflow_weights
            self._flows_probabs[self._induction_mask] = 1 - ((1 - self._induction * self._iflow_weights) ** pr).prod(axis=1)

            for st_i in range(len(self._stage_starts)):
                self._flows_probabs[self._outputs[:, st_i]] = self._correct_p(
                    self._flows_probabs[self._outputs[:, st_i]])
                self._flows_values[self._outputs[:, st_i]] = self._flows_probabs[self._outputs[:, st_i]] * pr[st_i]
            changes = self._flows_values @ self._targets - self._flows_values @ self._outputs
            self._result[step] = pr + changes

    def _determ_step(self, step: int):
        self._flows_probabs[self._induction_mask] = 1 - ((1 - self._induction * self._iflow_weights) **
                                                         self._result[step - 1]).prod(axis=1)

        for st_i in range(len(self._stage_starts)):
            self._flows_probabs[self._outputs[:, st_i]] = self._correct_p(self._flows_probabs[self._outputs[:, st_i]])
            self._flows_values[self._outputs[:, st_i]] = self._flows_probabs[self._outputs[:, st_i]] * \
                                                         self._result[step - 1][st_i]
        changes = self._flows_values @ self._targets - self._flows_values @ self._outputs
        self._result[step] = self._result[step - 1] + changes

    def _stoch_step(self, step: int):
        self._flows_probabs[self._induction_mask] = 1 - ((1 - self._induction * self._iflow_weights) **
                                                         self._result[step - 1]).prod(axis=1)

        for st_i in range(len(self._stage_starts)):
            self._flows_probabs[self._outputs[:, st_i]] = self._correct_p(self._flows_probabs[self._outputs[:, st_i]])
            flow_values = self._flows_probabs[self._outputs[:, st_i]] * self._result[step - 1][st_i]
            flow_values = poisson.rvs(flow_values, size=len(flow_values))
            flows_sum = flow_values.sum()
            if flows_sum > self._result[step - 1][st_i]:
                # находим избыток всех потоков ушедших из стадии st_i
                # распределим (вычтем) этот избыток из всех потоков пропорционально значениям потоков
                excess = flows_sum - self._result[step-1][st_i]
                flow_values = flow_values - flow_values / flows_sum * excess
            self._flows_values[self._outputs[:, st_i]] = flow_values
        changes = self._flows_values @ self._targets - self._flows_values @ self._outputs
        self._result[step] = self._result[step - 1] + changes

    def _save_additional_results(self, step: int):
        self._result_flows[step-1] = self._flows_values
        self._result_factors[step-1] = [fa.value for fa in self._factors]

    @classmethod
    def get_table(cls, table_df: pd.DataFrame) -> PrettyTable:
        table = PrettyTable()
        table.add_column('step', table_df.index.tolist())
        for col in table_df:
            table.add_column(col, table_df[col].tolist())
        table.float_format = f".{cls.__len_float}"
        return table

    def _get_result_df(self) -> pd.DataFrame:
        result = pd.DataFrame(self._result, columns=self._stage_names)
        return result.reindex(np.arange(self._duration), method='ffill')

    def _get_factors_df(self) -> pd.DataFrame:
        factors = pd.DataFrame(self._result_factors, columns=[fa.name for fa in self._factors])
        return factors.reindex(np.arange(self._duration))

    def _get_flows_df(self) -> pd.DataFrame:
        flows = pd.DataFrame(self._result_flows, columns=list(self._flow_names))
        return flows.reindex(np.arange(self._duration), fill_value=0)

    def _get_full_df(self) -> pd.DataFrame:
        return pd.concat([self._get_result_df(), self._get_flows_df(), self._get_factors_df()], axis=1)

    @property
    def result_df(self):
        return self._get_result_df()

    @property
    def full_df(self):
        return self._get_full_df()

    @property
    def flows_df(self):
        return self._get_flows_df()

    @property
    def factors_df(self):
        return self._get_factors_df()

    def print_result_table(self) -> None:
        print(self.get_table(self._get_result_df()))

    def print_factors_table(self) -> None:
        print(self.get_table(self._get_factors_df()))

    def print_flows_table(self) -> None:
        print(self.get_table(self._get_flows_df()))

    def print_full_result(self) -> None:
        print(self.get_table(self._get_full_df()))

    @property
    def name(self) -> str:
        return self._name

    def _write_table(self, filename: str, table: pd.DataFrame, floating_point='.', delimiter=',') -> None:
        table.to_csv(filename, sep=delimiter, decimal=floating_point,
                     float_format=f'%.{self.__len_float}f', index_label='step')

    def write_results(self, floating_point='.', delimiter=',', path: str = '',
                      write_flows: bool = False, write_factors: bool = False) -> None:

        if path and path[-1] != '\\':
            path = path + '\\'

        current_time = datetime.today().strftime('%d_%b_%y_%H-%M-%S')
        filename = f'{path}{self._name}_result_{current_time}.csv'

        tables = [self._get_result_df()]
        if write_flows:
            if self._result_flows is None:
                print('Warning: Results for flows should be saved while model is running')
            else:
                tables.append(self._get_flows_df())
        if write_factors:
            if self._result_factors is None:
                print('Warning: Results for factors should be saved while model is running')
            else:
                tables.append(self._get_factors_df())
        final_table = pd.concat(tables, axis=1)
        self._write_table(filename, final_table, floating_point, delimiter)

    def set_factors(self, **kwargs) -> None:
        for f in self._factors:
            if f.name in kwargs:
                f.set_fvalue(kwargs[f.name])

        self._dynamic_factors = [f for f in self._factors if f.is_dynamic]

    def set_start_stages(self, **kwargs) -> None:
        for s_index, s  in enumerate(self._stages):
            if s.name in kwargs:
                s.num = kwargs[s.name]
                self._stage_starts[s_index] = kwargs[s.name]


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

    def get_latex(self, simplified: bool = False) -> str:
        for fl in self._flows:
            fl.send_latex_terms(simplified)

        tab = '    '
        system_of_equations = f'\\begin{{equation}}\\label{{eq:{self._name}_{'classic' if simplified else 'full'}}}\n'
        system_of_equations += f'{tab}\\begin{{cases}}\n'

        for st in self._stages:
            system_of_equations += f'{tab * 2}{st.get_latex_equation()}\\\\\n'

        system_of_equations += f'{tab}\\end{{cases}}\n'
        system_of_equations += f'\\end{{equation}}\n'

        for st in self._stages:
            st.clear_latex_terms()

        return system_of_equations

    @staticmethod
    def _correct_p(probs: np.ndarray) -> np.ndarray:
        # матрица случившихся событий
        happened_matrix = np.array(list(product([0, 1], repeat=len(probs))), dtype=np.bool)

        # вектор вероятностей каждого сценария
        # те что свершились с исходной вероятностью, а не свершились - (1 - вероятность)
        full_probs = (probs * happened_matrix + (1 - probs) * (~happened_matrix)).prod(axis=1)

        # делим на то сколько событий произошло, в каждом сценарии
        # в первом случае ни одно событие не произошло, значит делить придётся на 0
        # а случай этот не пригодится
        full_probs[1:] = full_probs[1:] / happened_matrix[1:].sum(axis=1)

        # новые вероятности
        # по сути сумма вероятностей сценариев, в которых нужное событие произошло
        new_probs = np.zeros_like(probs)
        for i in range(len(probs)):
            new_probs[i] = full_probs[happened_matrix[:, i]].sum()
        return new_probs



