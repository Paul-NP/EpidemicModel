from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .flow import Flow

from itertools import product


class StageError(Exception):
    pass


class Stage:
    __MIN_NAME_LEN: int = 1
    __MAX_NAME_LEN: int = 20
    __FLOAT_LEN: int = 2

    __MIN_VALUE: int = 0

    @classmethod
    def _check_name(cls, name: str):
        if not isinstance(name, str):
            raise StageError('The stage name must be str')
        if len(name.split()) > 1:
            raise StageError('The stage name must be one word')
        if not cls.__MIN_NAME_LEN <= len(name) <= cls.__MAX_NAME_LEN:
            raise StageError(f'The stage name has an invalid length. Valid range '
                             f'[{cls.__MIN_NAME_LEN}, {cls.__MAX_NAME_LEN}]')

    @classmethod
    def _check_value(cls, value: int | float):
        if not isinstance(value, float | int):
            raise StageError("Stage start num must be number")
        if value < cls.__MIN_VALUE:
            raise StageError(f'Starting number in the stage cannot be less than {cls.__MIN_VALUE}')

    def __init__(self, name: str, start_num: int | float):
        self._check_name(name)
        self._check_value(start_num)

        self._name: str = name
        self._current_num: float = float(start_num)
        self._start_num: float = float(start_num)
        self._changes: list[float | int] = []
        self._probability_out: dict[Flow, float] = {}

    def _correct_probabilities(self):
        probs = list(self._probability_out.values())
        result = []
        for i, pr in enumerate(probs):
            pr_new = 0
            for happened in product([0, 1], repeat=len(probs)):
                if happened[i]:
                    to_pr = 1
                    for j, h in enumerate(happened):
                        if h:
                            to_pr *= probs[j]
                        else:
                            to_pr *= 1 - probs[j]
                    to_pr = to_pr / sum(happened)
                    pr_new += to_pr
            result.append(pr_new)
        self._probability_out = dict(zip(self._probability_out.keys(), probs))

    def add_probability_out(self, fl: Flow, prob: float):
        self._probability_out[fl] = prob

    def send_out_flow(self):
        self._correct_probabilities()
        for fl, pr in self._probability_out.items():
            fl.set_change_in(pr * self._current_num)

        self._probability_out = {}

    def add_change(self, change: float | int):
        self._changes.append(change)

    def apply_changes(self):
        self._current_num += sum(self._changes)
        self._changes = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def num(self) -> float:
        return self._current_num

    @num.setter
    def num(self, value: int):
        self._check_value(value)
        self._start_num = value
        self.reset_num()

    def reset_num(self):
        self._current_num = self._start_num

    def __str__(self) -> str:
        return f"Stage({self._name})"

    def __repr__(self) -> str:
        return f"Stage({self._name}) | {self._current_num:.{self.__FLOAT_LEN}f}"