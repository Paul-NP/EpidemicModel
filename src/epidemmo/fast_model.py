import numpy as np
from itertools import product
import time
import pandas as pd
from matplotlib import pyplot as plt
from .fast_flow import FastFlow
from .fast_stage import FastStage
from .fast_factor import FastFactor


from typing import Optional, Sequence


def correct_probabs(probs: np.ndarray) -> np.ndarray:
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


class FastEpidemicModel:
    def __init__(self, name: str, stages: list[FastStage], flows: list[FastFlow], relativity_factors: bool):
        self._name: str = name

        self._stage_names: dict[str, int] = {st.name: i for i, st in enumerate(stages)}
        self._stage_starts: np.ndarray = np.array([st.start_num for st in stages], dtype=np.float64)

        self._flows_weights: np.ndarray = np.zeros(len(flows), dtype=np.float64)
        self._targets: np.ndarray = np.zeros((len(flows), len(stages)), dtype=np.float64)
        self._induction_weights: np.ndarray = np.zeros((len(flows), len(stages)), dtype=np.float64)
        self._outputs: np.ndarray = np.zeros((len(flows), len(stages)), dtype=np.bool)

        self._connect_flows(flows)

        self._result: Optional[np.ndarray] = None

        self._flows_probabs: np.ndarray = np.zeros(len(flows), dtype=np.float64)
        self._flows_values: np.ndarray = np.zeros(len(flows), dtype=np.float64)

        self._induction_mask: np.ndarray = self._induction_weights.any(axis=1)
        self._induction: np.ndarray = self._induction_weights[self._induction_mask]

        self._flows_probabs[~self._induction_mask] = self._flows_weights[~self._induction_mask]

        self._relativity_factors: bool = relativity_factors

        if not self._relativity_factors:
            self._flows_weights[self._induction_mask] /= self._stage_starts.sum()

        print('стадии:\n', self._stage_names)
        print('матрица весов:\n', self._flows_weights)
        print('матрица индукции:\n', self._induction_weights)
        print('матрица целей:\n', self._targets)
        print('матрица выходов:\n', self._outputs)
        print('стартовые численности:\n', self._stage_starts)

    def _connect_flows(self, flows: list[FastFlow]):
        for fl in flows:
            fl.connect_matrix(self._flows_weights, self._targets, self._induction_weights, self._outputs)

    def run(self, duration: int):
        self._result = np.zeros((duration + 1, len(self._stage_starts)), dtype=np.float64)
        self._result[0] = self._stage_starts

        iflow_weights = self._flows_weights[self._induction_mask]
        for step in range(1, duration + 1):
            self._flows_probabs[self._induction_mask] = 1 - ((1 - self._induction * iflow_weights) **
                                                             self._result[step - 1]).prod(axis=1)

            for st_i in range(len(self._stage_starts)):
                self._flows_probabs[self._outputs[:, st_i]] = correct_probabs(self._flows_probabs[self._outputs[:, st_i]])
                self._flows_values[self._outputs[:, st_i]] = self._flows_probabs[self._outputs[:, st_i]] * self._result[step - 1][st_i]
            changes = self._flows_values @ self._targets - self._flows_values @ self._outputs
            self._result[step] = self._result[step - 1] + changes
        return pd.DataFrame(self._result, columns=list(self._stage_names.keys()))



def fast_sir(duration: int):
    # sir модель
    stage_numbers = np.array([100, 1, 0])
    flow_weights = np.array([0.004, 0.1])
    induction_weights = np.array([[0, 1, 0], [0, 0, 0]])
    induction_mask = induction_weights.any(axis=1)
    induction = induction_weights[induction_mask]
    targets = np.array([[0, 1, 0], [0, 0, 1]])
    outputs = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.bool)
    
    flows_probabs = np.array([0, 0], dtype=np.float64)
    flows_probabs[~induction_mask] = flow_weights[~induction_mask]
    flows_values = np.zeros_like(flows_probabs)
    
    result = np.zeros((duration + 1, len(stage_numbers)), dtype=np.float64)
    result[0] = stage_numbers

    start = time.time()
    for step in range(1, duration + 1):
        iflow_weights = flow_weights[induction_mask]
    
        flows_probabs[induction_mask] = 1 - ((1 - induction * iflow_weights) ** result[step - 1]).prod(axis=1)
        for st_i in range(len(stage_numbers)):
            flows_probabs[outputs[:, st_i]] = correct_probabs(flows_probabs[outputs[:, st_i]])
            flows_values[outputs[:, st_i]] = flows_probabs[outputs[:, st_i]] * result[step - 1][st_i]
        changes = flows_values @ targets - flows_values @ outputs
        result[step] = result[step - 1] + changes

    end = time.time()
    print(f'время выполнения матричного моделирования: {end - start}')
    return result


