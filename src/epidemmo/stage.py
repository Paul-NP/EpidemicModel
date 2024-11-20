from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .flow import Flow


class StageError(Exception):
    pass


class Stage:
    __MIN_NAME_LEN: int = 1
    __MAX_NAME_LEN: int = 20
    __FLOAT_LEN: int = 2

    __MIN_VALUE: int = 0

    @classmethod
    def _check_name(cls, name: str) -> None:
        if not isinstance(name, str):
            raise StageError('The stage name must be str')
        if len(name.split()) > 1:
            raise StageError('The stage name must be one word')
        if not cls.__MIN_NAME_LEN <= len(name) <= cls.__MAX_NAME_LEN:
            raise StageError(f'The stage name has an invalid length. Valid range '
                             f'[{cls.__MIN_NAME_LEN}, {cls.__MAX_NAME_LEN}]')

    @classmethod
    def _check_value(cls, value: int | float) -> None:
        if not isinstance(value, float | int):
            raise StageError("Stage start num must be number")
        if value < cls.__MIN_VALUE:
            raise StageError(f'Starting number in the stage cannot be less than {cls.__MIN_VALUE}')

    def __init__(self, name: str, start_num: int | float, *, index: int) -> None:
        self._check_name(name)
        self._check_value(start_num)

        self._name: str = name
        self._start_num: float = float(start_num)

        self._latex_repr: Optional[str] = None
        self._latex_outs: list[str] = []
        self._latex_inputs: list[str] = []

        self._index: int = index

    @property
    def index(self) -> int:
        return self._index

    @property
    def name(self) -> str:
        return self._name

    @property
    def start_num(self) -> float:
        return self._start_num

    @start_num.setter
    def start_num(self, value: int) -> None:
        self._check_value(value)
        self._start_num = value

    def __str__(self) -> str:
        return f"Stage({self._name})"

    def __repr__(self) -> str:
        return f"Stage({self._name}, {self._start_num:.{self.__FLOAT_LEN}f})"

    def add_latex_out(self, term: str) -> None:
        self._latex_outs.append(term)

    def add_latex_input(self, term: str) -> None:
        self._latex_inputs.append(term)

    def set_latext_repr(self, latex_repr: Optional[str]) -> None:
        self._latex_repr = latex_repr

    def get_latex_repr(self) -> str:
        if self._latex_repr is None:
            return self._name
        return self._latex_repr

    def get_latex_equation(self) -> str:
        positive_terms = '+'.join(self._latex_inputs)
        negative_terms = '-'.join(self._latex_outs)
        if negative_terms:
            negative_terms = f' - {negative_terms}'
        summa = positive_terms + negative_terms

        equation = f'\\frac{{d{self.get_latex_repr()}}}{{dt}} = {summa}'
        return equation

    def clear_latex_terms(self):
        self._latex_inputs.clear()
        self._latex_outs.clear()



