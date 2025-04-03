from typing import Optional, Callable

import numpy as np

import pandas as pd
from itertools import product

from matplotlib import pyplot as plt
from prettytable import PrettyTable
from scipy.stats import poisson, alpha
from datetime import datetime

from .flow import Flow
from .stage import Stage
from .factor import Factor


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

        self._delta: int = 1

        # факторы, которые будут изменяться во время моделирования
        self._dynamic_factors: list[Factor] = [fa for fa in self._factors if fa.is_dynamic]

        self._flows_weights: np.ndarray = np.zeros(len(flows), dtype=np.float64)
        self._targets: np.ndarray = np.zeros((len(flows), len(stages)), dtype=np.float64)
        self._induction_weights: np.ndarray = np.zeros((len(flows), len(stages)), dtype=np.float64)
        self._outputs: np.ndarray = np.zeros((len(flows), len(stages)), dtype=np.bool_)

        # связываем факторы, используемые в потоках, с матрицами
        self._connect_matrix(flows)

        self._duration = 1
        self._result: np.ndarray = np.zeros((self._duration, len(stages)), dtype=np.float64)
        self._result[0] = self._stage_starts
        self._result_flows: Optional[np.ndarray] = None
        self._result_factors: Optional[np.ndarray] = None

        self._peaks: Optional[np.ndarray] = None

        self._cis_significance = 0.05
        self._confidence: Optional[np.ndarray] = None
        self._confidence_peaks: Optional[np.ndarray] = None
        self._confidence_flows: Optional[np.ndarray] = None

        self._flows_probabs: np.ndarray = np.zeros(len(flows), dtype=np.float64)
        self._flows_values: np.ndarray = np.zeros(len(flows), dtype=np.float64)
        self._induction_mask: np.ndarray = self._induction_weights.any(axis=1)
        self._induction: np.ndarray = self._induction_weights[self._induction_mask]

        self._relativity_factors: bool = False
        self.set_relativity_factors(relativity_factors)

    @property
    def population_size(self) -> int:
        """
        :return: Размер популяции
        """
        return self._stage_starts.sum()

    def set_relativity_factors(self, relativity_factors: bool):
        """
        :param relativity_factors: относительные ли факторы (относительные - не будут делиться на размер популяции)
        :return:
        """
        if not isinstance(relativity_factors, bool):
            raise ModelError('relativity_factors must be bool')
        for fl in self._flows:
            fl.set_relativity_factors(relativity_factors)
        self._relativity_factors = relativity_factors

    def _update_all_factors(self):
        for fa in self._factors:
            fa.update(0)

    def _update_dynamic_factors(self, step: int, *args):
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
        self._flows_weights[self._induction_mask] /= self.population_size

    def _connect_matrix(self, flows: list[Flow]):
        for fl in flows:
            fl.connect_matrix(self._flows_weights, self._targets, self._induction_weights, self._outputs)

    def _prepare(self):
        self._update_all_factors()
        if not self._relativity_factors:
            self._correct_not_rel_factors()
        self._prepare_factors_matrix()

    def start(self, duration: int, *, full_save: bool = False, stochastic: bool = False, delta: int = 1,
              get_cis: bool = False, num_cis_starts: int = 100, cis_significance: float = 0.05,
              align_results_by_peaks: bool = True) -> pd.DataFrame:
        """
        Запускает модель и возвращает таблицу результатов
        :param duration: длительность моделирования
        :param full_save: вычислить ли все результаты (+потоки, +факторы)
        :param stochastic: запускать ли модель в стохастическом режиме
        :param delta: шаг времени для представления результатов
        :param get_cis: вычислить ли доверительные интервалы
        :param num_cis_starts: количество стохастических запусков для вычисления доверительных интервалов
        :param cis_significance: уровень значимости для доверительных интервалов
        :param align_results_by_peaks: выравнивать ли результаты стохастических запусков по моменту пика при вычислении доверительных интервалов
        :return: таблица результатов, столбцы - стадии, строки - шаги моделирования
        """
        if not isinstance(duration, int) or duration < 1:
            raise ModelError('duration must be int > 1')
        if not isinstance(full_save, bool):
            raise ModelError('full_save must be bool')
        if not isinstance(stochastic, bool):
            raise ModelError('stochastic must be bool')
        if not isinstance(cis_significance, float) or not 0 < cis_significance < 1:
            raise ModelError('cis_significance must be float in (0, 1)')
        if not isinstance(delta, int) or not delta >= 1:
            raise ModelError('delta must be int >= 1')

        duration = (duration - 1) * delta + 1

        self._result_flows = None
        self._result_factors = None
        self._confidence = None
        self._confidence_flows = None
        self._peaks = None
        self._confidence_peaks = None
        self._duration = duration
        self._cis_significance = cis_significance
        self._delta = delta

        if get_cis:
            self._get_confidence_intervals(num_cis_starts, full_save, cis_significance, align_results_by_peaks)

        self._start(full_save, stochastic)
        return self._get_result_df()

    def _start(self, save_full: bool, stochastic: bool):
        self._result = np.zeros((self._duration, len(self._stage_starts)), dtype=np.float64)
        self._result[0] = self._stage_starts

        self._prepare()

        if not self._dynamic_factors and not save_full and not stochastic:
            self._fast_run()
            self._peaks = self._get_teor_peaks(self._result)
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

        self._full_step_seq.append(self._result_shift)

        if save_full:
            self._full_step_seq.append(self._save_additional_results)
            self._result_flows = np.zeros((self._duration, len(self._flow_names)), dtype=np.float64)
            self._result_factors = np.full((self._duration, len(self._factors_names)), np.nan, dtype=np.float64)
        else:
            self._result_flows = None
            self._result_factors = None

        if stochastic:
            self._stoch_run()
        else:
            self._determ_run()
            self._peaks = self._get_teor_peaks(self._result)

        if save_full:
            self._result_flows = self._result_flows[1:]
            self._result_factors = self._result_factors[1:]

    def _several_stoch_starts(self, count: int, flows_cis: bool = False) -> tuple[np.ndarray, np.ndarray]:
        all_results = np.zeros((count, self._duration, len(self._stage_starts)), dtype=np.float64)
        all_results[:, 0, :] = self._stage_starts
        flows_results = None
        if flows_cis:
            flows_results = np.full((count, self._duration, len(self._flow_names)), -1, dtype=np.float64)
            # flows_results = np.zeros((count, self._duration, len(self._flow_names)), dtype=np.float64)

        self._full_step_seq: list[Callable] = []
        if self._dynamic_factors:
            self._full_step_seq.append(self._update_dynamic_factors)
            if not self._relativity_factors:
                self._full_step_seq.append(self._correct_not_rel_factors)
            self._full_step_seq.append(self._prepare_factors_matrix)

        self._full_step_seq.append(self._stoch_step)
        self._full_step_seq.append(self._result_shift)

        if flows_cis:
            self._full_step_seq.append(self._save_flows_results)

        for i in range(count):
            if i % 100 == 0 and i > 0:
                print(f'Совершено {i} стохастических запусков.')
            self._result = all_results[i]
            if flows_cis:
                self._result_flows = flows_results[i]
            self._prepare()
            self._stoch_run()
        print(f'Совершено {count} стохастических запусков.')
        if flows_cis:
            flows_results = flows_results[:, 1:, :]

        return all_results, flows_results

    def get_stoch_variants(self, duration: int, num_starts: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Результаты по стадиям и по потокам для множества стохастических запусков в виде трёхмерных матриц
        :param duration: продолжительность моделирования
        :param num_starts: количество стохастических запусков
        :return: матрица с результатами по стадиям, матрица с результатами по потокам
        """
        if not isinstance(duration, int) or duration < 1:
            raise ModelError('duration must be int > 1')
        if not isinstance(num_starts, int) or num_starts < 1:
            raise ModelError('Number of starts for calculating confidence intervals must be int > 1')

        self._duration = duration
        return self._several_stoch_starts(num_starts, True)

    def _get_confidence_intervals(self, num_starts: int, flows_cis: bool, alpha_cis: float, align: bool):
        if not isinstance(num_starts, int) or num_starts < 1:
            raise ModelError('Number of starts for calculating confidence intervals must be int > 1')

        results, flows_results = self._several_stoch_starts(num_starts, flows_cis)
        self._start(flows_cis, False)  # для получения теоретического результата в self._result

        results = self._first_by_delta_results(results)
        result = self._first_by_delta(self._result)
        shifts = self._calc_shifts(results, result)
        if align:
            results = self._align_stoch_results(results, shifts)
        self._confidence = self._calc_confidence_intervals(results, result, alpha_cis)
        self._confidence_peaks = self._calc_confidence_peak_interval(result, shifts, alpha_cis)


        if flows_cis:
            # потоки не выравниваю и не считаем доверительный интервал для пиков
            flows_results = self._sum_by_delta_results(flows_results)
            flows_result = self._sum_by_delta(self._result_flows)
            self._confidence_flows = self._calc_confidence_intervals(flows_results, flows_result, alpha_cis)

    def _calc_shifts(self, stoch_results: np.array, teor_result: np.array) -> np.array:
        diffs_sign = np.sign(np.diff(teor_result, axis=0))
        with_peaks = np.any(diffs_sign > 0, axis=0) & np.any(diffs_sign < 0, axis=0)
        if not np.any(with_peaks):
            print('Warning: не найдено ни одного состояния с пиком. Доверительные интервалы оценены без выравнивания.')
            return np.zeros(stoch_results.shape[0])

        increasings = teor_result[0, :] < teor_result.max(axis=0)
        incr_peaks = increasings & with_peaks  # линии которые возрастают и имеют пик - больше всего похоже на I
        if np.any(increasings & with_peaks):
            estimating_line_i = np.where(incr_peaks)[0][0]  # первая линия из лучших похожих
            peak_pos = teor_result[:, estimating_line_i].argmax()
        else:
            estimating_line_i = np.where(with_peaks)[0][0] # первая линия из тех кто просто с пиком (минимумом)
            peak_pos = teor_result[:, estimating_line_i].argmin()

        print(f'Динамика численности "{self._stage_names[estimating_line_i]}" будет '
              f'рассматриваться для поиска времени пика')

        shifts = peak_pos - stoch_results[:, :, estimating_line_i].argmax(axis=1)
        return shifts

    @staticmethod
    def _align_stoch_results(stoch_results: np.array, shifts: np.array):
        num_runs, duration, num_lines = stoch_results.shape
        aligned_stoch_results = np.full((num_runs, duration, num_lines), np.nan, dtype=np.float64)
        for st_i in range(num_lines):
            for run_i in range(num_runs):
                shift = shifts[run_i]
                start = shift if shift > 0 else 0
                end = duration + shift if shift < 0 else duration
                rolled = np.roll(stoch_results[run_i, :, st_i], shift, axis=0)
                aligned_stoch_results[run_i, start:end, st_i] = rolled[start:end]
        return aligned_stoch_results

    def _get_teor_peaks(self, teor_result: np.array) -> np.array:
        diffs_sign = np.sign(np.diff(teor_result, axis=0))
        with_peaks = np.any(diffs_sign > 0, axis=0) & np.any(diffs_sign < 0, axis=0)
        increasings = teor_result[0, :] < teor_result.max(axis=0)
        teor_peaks = np.full(teor_result.shape[1], np.nan, dtype=np.float64)
        teor_peaks[with_peaks & increasings] = teor_result[:, with_peaks & increasings].argmax(axis=0)
        teor_peaks[with_peaks & ~increasings] = teor_result[:, with_peaks & ~increasings].argmin(axis=0)
        return teor_peaks

    def _calc_confidence_peak_interval(self, teor_result: np.array, shifts: np.array, alpha: float):
        peaks = self._get_teor_peaks(teor_result)
        down_limit = alpha * 100
        up_limit = (1 - alpha) * 100

        shift_limits = np.percentile(shifts, [down_limit, up_limit], axis=0)
        confidence = np.full((2, teor_result.shape[1]), np.nan, dtype=np.float64)

        diffs_sign = np.sign(np.diff(teor_result, axis=0))
        with_peaks = np.any(diffs_sign > 0, axis=0) & np.any(diffs_sign < 0, axis=0)
        with_peaks = np.where(with_peaks)[0]

        for st_i in with_peaks:
            left_bound = peaks[st_i] + shift_limits[0]
            right_bound = peaks[st_i] + shift_limits[1]
            confidence[0, st_i] = left_bound
            confidence[1, st_i] = right_bound

        return confidence

    @staticmethod
    def _calc_confidence_intervals(stoch_results: np.array, mid_result: np.array, alpha_cis: float = 0.05):
        down_limit = alpha_cis * 100
        up_limit = (1 - alpha_cis) * 100
        num_runs, duration, num_stages = stoch_results.shape
        confidence = np.zeros((duration, num_stages * 2))
        for st_i in range(num_stages):
            for time in range(duration):
                low_results_mask = (stoch_results[:, time, st_i] < mid_result[time, st_i]) & (stoch_results[:, time, st_i] > -1)
                low_results = stoch_results[low_results_mask, time, st_i]
                if len(low_results):
                    conf_low = np.percentile(low_results, [down_limit], axis=0)[0]
                else:
                    conf_low = mid_result[time, st_i]

                up_results = stoch_results[stoch_results[:, time, st_i] > mid_result[time, st_i], time, st_i]
                if len(up_results):
                    conf_up = np.percentile(up_results, [up_limit], axis=0)[0]
                else:
                    conf_up = mid_result[time, st_i]

                confidence[time, st_i * 2: st_i * 2 + 2] = [conf_low, conf_up]
        return confidence

    def _determ_run(self):
        self._result_shift(0, 1)
        for step in range(1, self._duration):
            for step_func in self._full_step_seq:
                step_func(step, 1)
            # self._result_shift(step, 1)

    def _stoch_run(self):
        self._result_shift(0, 1)
        step = 1
        while step < self._duration:
            shift = poisson.rvs(mu=1)
            for step_func in self._full_step_seq:
                step_func(step, shift)
            step += shift

    def _result_shift(self, step: int, shift):
        self._result[step: step + shift + 1] = self._result[step]

    def _fast_run(self):
        for step in range(1, self._duration):
            pr = self._result[step - 1]
            self._induction * self._iflow_weights
            self._flows_probabs[self._induction_mask] = 1 - ((1 - self._induction * self._iflow_weights) ** pr).prod(axis=1)

            for st_i in range(len(self._stage_starts)):
                self._flows_probabs[self._outputs[:, st_i]] = self._correct_p(
                    self._flows_probabs[self._outputs[:, st_i]])
                self._flows_values[self._outputs[:, st_i]] = self._flows_probabs[self._outputs[:, st_i]] * pr[st_i]
            changes = self._flows_values @ self._targets - self._flows_values @ self._outputs
            self._result[step] = pr + changes

    def _determ_step(self, step: int, *args):
        self._flows_probabs[self._induction_mask] = 1 - ((1 - self._induction * self._iflow_weights) **
                                                         self._result[step]).prod(axis=1)

        for st_i in range(len(self._stage_starts)):
            self._flows_probabs[self._outputs[:, st_i]] = self._correct_p(self._flows_probabs[self._outputs[:, st_i]])
            self._flows_values[self._outputs[:, st_i]] = self._flows_probabs[self._outputs[:, st_i]] * \
                                                         self._result[step][st_i]
        changes = self._flows_values @ self._targets - self._flows_values @ self._outputs
        self._result[step] += changes

    def _stoch_step(self, step: int, *args):
        self._flows_probabs[self._induction_mask] = 1 - ((1 - self._induction * self._iflow_weights) **
                                                         self._result[step]).prod(axis=1)

        for st_i in range(len(self._stage_starts)):
            self._flows_probabs[self._outputs[:, st_i]] = self._correct_p(self._flows_probabs[self._outputs[:, st_i]])
            flow_values = self._flows_probabs[self._outputs[:, st_i]] * self._result[step][st_i]
            flow_values = poisson.rvs(flow_values, size=len(flow_values))
            flows_sum = flow_values.sum()
            if flows_sum > self._result[step][st_i]:
                # находим избыток всех потоков ушедших из стадии st_i
                # распределим (вычтем) этот избыток из всех потоков пропорционально значениям потоков
                excess = flows_sum - self._result[step][st_i]
                flow_values = flow_values - flow_values / flows_sum * excess
            self._flows_values[self._outputs[:, st_i]] = flow_values
        changes = self._flows_values @ self._targets - self._flows_values @ self._outputs
        self._result[step] += changes

    def _save_additional_results(self, step: int, *args):
        self._result_flows[step] += self._flows_values
        self._result_factors[step] = [fa.value for fa in self._factors]

    def _save_flows_results(self, step: int, *args):
        self._result_flows[step] += self._flows_values

    @classmethod
    def _get_table(cls, table_df: pd.DataFrame) -> PrettyTable:
        table = PrettyTable()
        table.add_column('step', table_df.index.tolist())
        for col in table_df:
            col_name = f'{col[0]}_{col[1]}' if isinstance(col, tuple) else str(col)
            table.add_column(col_name, table_df[col].tolist())
        table.float_format = f".{cls.__len_float}"
        return table

    def _get_result_df(self) -> pd.DataFrame:
        result = pd.DataFrame(self._first_by_delta(self._result), columns=self._stage_names)
        return result.reindex(np.arange(len(result)), method='ffill')

    def _get_factors_df(self) -> pd.DataFrame:
        if self._result_factors is None:
            print('Warning: результаты по факторам должны быть сохранены во время моделирования')
            return pd.DataFrame()
        return pd.DataFrame(self._first_by_delta(self._result_factors), columns=[fa.name for fa in self._factors])

    def _get_flows_df(self) -> pd.DataFrame:
        if self._result_flows is None:
            print('Warning: результаты по потокам должны быть сохранены во время моделирования')
            return pd.DataFrame()
        flows = pd.DataFrame(self._sum_by_delta(self._result_flows), columns=self._flow_names)
        flows.fillna(0, inplace=True)
        return flows

    def _get_full_df(self) -> pd.DataFrame:
        a = self._get_result_df()
        b = self._get_conf_df()
        c = self._get_flows_df()
        d = self._get_conf_flows_df()
        e = self._get_factors_df()
        return pd.concat([self._get_result_df(), self._get_conf_df(), self._get_flows_df(), self._get_conf_flows_df(),
                          self._get_factors_df()], axis=1)

    def _get_conf_df(self):
        if self._confidence is None:
            print('Warning: результаты по доверительным интервалам стадий должны быть сохранены во время моделирования')
            return pd.DataFrame()
        col_names = [(st_name, limit) for st_name in self._stage_names for limit in ['lower', 'upper']]
        index = pd.MultiIndex.from_tuples(col_names, names=['stage', 'limit'])
        conf_df = pd.DataFrame(self._confidence, columns=index)
        return conf_df.reindex(np.arange(len(conf_df)))

    def _get_conf_flows_df(self):
        if self._confidence_flows is None:
            print('Warning: результаты по доверительным интервалам потоков должны быть сохранены во время моделирования')
            return pd.DataFrame()
        col_names = [(fl_name, limit) for fl_name in self._flow_names for limit in ['lower', 'upper']]
        index = pd.MultiIndex.from_tuples(col_names, names=['flow', 'limit'])
        conf_df = pd.DataFrame(self._confidence_flows, columns=index)
        return conf_df

    def _get_conf_peaks_df(self):
        if self._confidence_peaks is None:
            print('Warning: результаты по доверительным интервалам стадий должны быть сохранены во время моделирования')
            return pd.DataFrame()
        conf_df = pd.DataFrame(self._confidence_peaks, columns=self._stage_names, index=['lower', 'upper'])
        return conf_df

    def _get_peaks_df(self):
        if self._peaks is None:
            print('Warning: результаты моментов пиков численности будут только при детерменированном запуске и при оценке доверительных интервалов')
            return pd.DataFrame()

        peaks_data = np.full((2, len(self._stage_names)), np.nan)
        with_peak = ~ np.isnan(self._peaks)
        with_peak_results = self._result[:, with_peak]
        num_peaks = with_peak.sum()
        peaks_data[0, with_peak] = self._peaks[with_peak]
        peaks_data[1, with_peak] = with_peak_results[self._peaks[with_peak].astype(np.int64), np.arange(num_peaks)]

        return pd.DataFrame(peaks_data, columns=self._stage_names, index=['moment', 'value'])

    @property
    def result_df(self) -> pd.DataFrame:
        """
        :return: Таблица результатов, численность каждой стадии во времени
        """
        return self._get_result_df()

    @property
    def full_df(self) -> pd.DataFrame:
        """
        :return: Таблица результатов, численность каждой стадии, потока и фактора во времени
        """
        return self._get_full_df()

    @property
    def flows_df(self) -> pd.DataFrame:
        """
        :return: Таблица потоков, интенсивность (численность) каждого потока (перехода) во времени
        """
        return self._get_flows_df()

    @property
    def factors_df(self):
        """
        :return: Таблица факторов (параметров модели), значение каждого фактора во времени
        """
        return self._get_factors_df()

    @property
    def peaks_df(self):
        """
        :return: Таблица пиков (моментов и значений) численности каждой стадии - первого локального максимума / минимума
        """
        return self._get_peaks_df()


    @property
    def confidence_df(self):
        """
        :return: Таблица доверительных интервалов, верхняя и нижняя границы для численности каждой стадии во времени
        """
        return self._get_conf_df()

    @property
    def confidence_peaks_df(self):
        """
        :return: Таблица доверительных интервалов, левый и правый моменты возможного пика численности каждой стадии
        """
        return self._get_conf_peaks_df()

    @property
    def confidence_flows_df(self):
        """
        :return: Таблица доверительных интервалов, верхняя и нижняя границы для интенсивности каждого потока во времени
        """
        return self._get_conf_flows_df()

    def print_result_table(self) -> None:
        print(self._get_table(self._get_result_df()))

    def print_factors_table(self) -> None:
        print(self._get_table(self._get_factors_df()))

    def print_flows_table(self) -> None:
        print(self._get_table(self._get_flows_df()))

    def print_full_table(self) -> None:
        print(self._get_table(self._get_full_df()))

    @property
    def name(self) -> str:
        """
        :return: Название модели
        """
        return self._name

    def _write_table(self, filename: str, table: pd.DataFrame, floating_point='.', delimiter=',') -> None:
        table.to_csv(filename, sep=delimiter, decimal=floating_point,
                     float_format=f'%.{self.__len_float}f', index_label='step')

    def write_results(self, floating_point='.', delimiter=',', path: str = '',
                      write_flows: bool = False, write_factors: bool = False) -> None:
        """
        Сохраняет результаты модели в csv-файл
        :param floating_point: десятичная точка
        :param delimiter: разделитель в таблице
        :param path: путь для сохранения
        :param write_flows: сохранять ли столбцы с результатами по потокам
        :param write_factors: сохранять ли столбцы с результатами по факторам
        """
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
                s.start_num = kwargs[s.name]
                self._stage_starts[s_index] = kwargs[s.name]

    def __str__(self) -> str:
        return f'Model({self._name})'

    def __repr__(self) -> str:
        return f'Model({self._name}): {list(self._flows)}'

    @property
    def stages(self) -> list[dict[str, float]]:
        return [{'name': st.name, 'num': float(st.start_num)} for st in self._stages]

    @property
    def stages_dict(self) -> dict[str, float]:
        return {st.name: float(st.start_num) for st in self._stages}

    @property
    def stage_names(self) -> list[str]:
        return list(self._stage_names)

    @property
    def factors(self) -> list[dict[str, float]]:
        return [{'name': fa.name, 'value': 'dynamic' if fa.is_dynamic else fa.value} for fa in self._factors]

    @property
    def factors_dict(self) -> dict[str, float | str]:
        return {fa.name: 'dynamic' if fa.is_dynamic else fa.value for fa in self._factors}

    @property
    def factor_names(self) -> list[str]:
        return list(self._factors_names)

    @property
    def flows(self) -> list[dict]:
        flows = []
        for fl in self._flows:
            fl_dict = {'start': fl.start.name, 'factor': fl.factor.name,
                       'end': {st.name: fa.name for st, fa in fl.ends.items()},
                       'inducing': {st.name: fa.name for st, fa in fl.inducing.items()}}
            flows.append(fl_dict)
        return flows

    @property
    def flow_names(self) -> list[str]:
        return list(self._flow_names)

    def get_latex(self, simplified: bool = False) -> str:
        for fl in self._flows:
            fl.send_latex_terms(simplified)

        tab = '    '
        system_of_equations = f"\\begin{{equation}}\\label{{eq:{self._name}_{'classic' if simplified else 'full'}}}\n"
        system_of_equations += f'{tab}\\begin{{cases}}\n'

        for st in self._stages:
            system_of_equations += f'{tab * 2}{st.get_latex_equation()}\\\\\n'

        system_of_equations += f'{tab}\\end{{cases}}\n'
        system_of_equations += f'\\end{{equation}}\n'

        for st in self._stages:
            st.clear_latex_terms()

        return system_of_equations

    def _prepare_plot_args(self, ax, draw_cis, draw_peaks) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        elif not isinstance(ax, plt.Axes):
            raise ModelError('ax must be plt.Axes')
        if not isinstance(draw_cis, bool):
            raise ModelError('draw_cis must be bool')
        if not isinstance(draw_peaks, bool):
            raise ModelError('draw_peaks must be bool')
        return ax

    def plot(self, *args, ax: Optional[plt.Axes] = None,
             draw_cis: bool = False, draw_peaks: bool = False) -> plt.Axes:
        """
        Построение графика изменений численности стадий.
        :param args: Названия стадий, которые будут включены в график.
        :param ax: Axes на котором будет построен график.
        :param draw_cis: Рисовать ли доверительные интервалы.
        :param draw_peaks: Рисовать ли доверительные интервалы пиков.
        :return: Axes на котором будет построен график.
        """
        ax = self._prepare_plot_args(ax, draw_cis, draw_peaks)

        stage_names = [s for s in args if s in self._stage_names]
        if not stage_names:
            stage_names = list(self._stage_names)

        if draw_cis:
            conf_df = self._get_conf_df()
            conf_df = conf_df[stage_names] if len(conf_df) > 0 else None
        else:
            conf_df = None

        if draw_peaks:
            peaks_df = self._get_peaks_df()
            peaks_conf_df = self._get_conf_peaks_df()
            if len(peaks_df) == 0 or len(peaks_conf_df) == 0:
                peaks_conf_df = None
                peaks_df = None
        else:
            peaks_conf_df = None
            peaks_df = None
        significance = (1 - self._cis_significance) * 100
        colors, _ = self._get_color_schema()
        return self._plot(ax=ax, df=self._get_result_df()[stage_names], conf_df=conf_df,
                          peaks_df=peaks_df, peaks_conf_df=peaks_conf_df,
                          ylabel='количество индивидов', name=self._name, significance=significance, colors=colors)

    def plot_flows(self, *args, ax: plt.Axes = None,
                   draw_cis: bool = True) -> plt.Axes:
        """
        Построение графика изменения интенсивности потоков.
        :param args: Названия потоков, которые будут включены в график.
        :param ax: Axes на котором будет построен график.
        :param draw_cis: Рисовать ли доверительные интервалы.
        :return: Axes на котором будет построен график.
        """
        ax = self._prepare_plot_args(ax, draw_cis, False)

        flow_names = [f for f in args if f in self._flow_names]
        if not flow_names:
            flow_names = list(self._flow_names)

        if draw_cis:
            conf_df = self._get_conf_flows_df()
            conf_df = conf_df[flow_names] if len(conf_df) > 0 else None
        else:
            conf_df = None

        significance = (1 - self._cis_significance) * 100
        _, colors = self._get_color_schema()
        return self._plot(ax=ax, df=self._get_flows_df()[flow_names], conf_df=conf_df, peaks_df=None, peaks_conf_df=None,
                          ylabel='интенсивность потока', name=f'потоки модели {self._name}', significance=significance,
                          colors=colors)

    @staticmethod
    def _plot(*, ax: Optional[plt.Axes], df: pd.DataFrame, conf_df: Optional[pd.DataFrame],
              peaks_df: Optional[pd.DataFrame], peaks_conf_df: Optional[pd.DataFrame],
              ylabel: str, name: str, significance: float, colors: dict) -> plt.Axes:

        labels = {sname: f'Прогноз «{sname}»,' for sname in df.columns}

        if conf_df is not None:
            for stage in df.columns:
                labels[stage] = f'{labels[stage]} доверительный интервал ({significance:.0f}%),\n'
                ax.fill_between(conf_df.index, conf_df[(stage, 'lower')], conf_df[(stage, 'upper')],
                                color=colors[stage], alpha=0.2)

        if peaks_conf_df is not None and peaks_df is not None:
            peak_line_height = df.to_numpy().max() / 20
            for stage in peaks_conf_df.columns:
                if not np.isnan(peaks_conf_df[stage]).any() and not np.isnan(peaks_df[stage]).any():
                    labels[stage] = f'{labels[stage]}доверительный интервал времени пика ({significance:.0f}%)'
                    x = list(peaks_conf_df[stage].astype(int))
                    y = peaks_df.loc['value', stage]
                    half_peak_line = peak_line_height / 2
                    ax.vlines(x, y - half_peak_line, y + half_peak_line, color=colors[stage], alpha=0.8)
                    ax.hlines(y, x[0], x[1], color=colors[stage], linestyle='--', alpha=0.8)

        for sname in df.columns:
            p = ax.plot(df[sname], label=labels[sname].strip().strip(','), color=colors[sname])

        ax.set_title(name)
        ax.set_xlabel('Время')
        ax.set_ylabel(ylabel)
        ax.grid()
        ax.legend()

        return ax


    @staticmethod
    def _correct_p(probs: np.ndarray) -> np.ndarray:
        # return probs
        # матрица случившихся событий
        happened_matrix = np.array(list(product([0, 1], repeat=len(probs))), dtype=np.bool_)

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

    def _first_by_delta(self, result: np.ndarray):
        return result[::self._delta, :]

    def _first_by_delta_results(self, results: np.ndarray):
        return results[:, ::self._delta, :]

    def _sum_by_delta(self, result: np.ndarray):
        rows, cols = result.shape
        temp = result.reshape(-1, self._delta, cols)
        return temp.sum(axis=1)

    def _sum_by_delta_results(self, results: np.ndarray):
        nums, rows, cols = results.shape
        temp = results.reshape(nums, -1, self._delta, cols)
        return temp.sum(axis=2)

    def _get_color_schema(self):
        standard_stages = {'S', 'I', 'R', 'E', 'D'}
        standard_flows = {'Flow(S>I)', 'Flow(S>E)', 'Flow(E>I)', 'Flow(I>R)', 'Flow(I>D)',
                          'Flow(I>R,D)', 'Flow(I>D,R)', 'Flow(R>S)'}
        '''
        S	#3914AF
        E	#FFD300
        I	#FF0000
        R	#00CC00
        D	#535353

        Flow(S>I)	#FF0000
        Flow(S>E)	#FFD300
        Flow(E>I)	#FF0000
        Flow(I>R)	#00CC00
        Flow(I>D)	#535353
        Flow(I>R,D)	#00CC00
        Flow(I>D,R)	#00CC00
        Flow(R>S)	#3914AF
        '''

        if set(self.stage_names).issubset(standard_stages) and set(self.flow_names).issubset(standard_flows):
            stage_colors = {'S': '#3914AF', 'E': '#FFD300', 'I': '#FF0000', 'R': '#00CC00', 'D': '#535353'}
            flow_colors = {'Flow(S>I)': '#FF0000', 'Flow(S>E)': '#FFD300', 'Flow(E>I)': '#FF0000',
                           'Flow(I>R)': '#00CC00', 'Flow(I>D)': '#535353', 'Flow(I>R,D)': '#00CC00',
                           'Flow(I>D,R)': '#00CC00', 'Flow(R>S)': '#3914AF'}
        else:
            cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            stage_colors = {self._stage_names[i]: cycle[i % len(cycle)] for i in range(len(self._stage_names))}
            flow_colors = {self._flow_names[i]: cycle[i % len(cycle)] for i in range(len(self._flow_names))}

        return stage_colors, flow_colors


