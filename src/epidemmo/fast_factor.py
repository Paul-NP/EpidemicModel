from __future__ import annotations
from typing import Callable, Optional, Literal, Any
from types import FunctionType

from numpy.typing import NDArray


class FactorError(Exception):
    pass


class FastFactor:
    __MIN_NAME_LEN: int = 1
    __MAX_NAME_LEN: int = 30

    @classmethod
    def __check_name(cls, name: str) -> None:
        if not isinstance(name, str):
            raise FactorError('The factor name must be str')
        if len(name.split()) > 1:
            raise FactorError('The factor name must be one word')
        if not cls.__MIN_NAME_LEN <= len(name) <= cls.__MAX_NAME_LEN:
            raise FactorError(f'The factor name "{name}" has an invalid length. Valid range '
                              f'[{cls.__MIN_NAME_LEN}, {cls.__MAX_NAME_LEN}]')

    def __init__(self, name: str, value: int | float | Callable[[int], float]) -> None:
        self.__check_name(name)

        self._name: str = name
        self._value: float = 0
        self._func: Optional[Callable[[int], float]] = None
        self._latex_repr: Optional[str] = None

        self.set_fvalue(value)

        self._connected_matrix: Optional[NDArray] = None
        self._value_pos_in_matrix: Optional[int | tuple[int, int]] = None

    def connect_matrix(self, matrix: NDArray, value_pos_in_matrix: int | tuple[int, int]) -> None:
        self._connected_matrix = matrix
        self._value_pos_in_matrix = value_pos_in_matrix
        self._connected_matrix[self._value_pos_in_matrix] = self._value

    def set_fvalue(self, value: int | float | Callable[[int], float]) -> None:
        match value:
            case int(value) | float(value):
                self._value = float(value)
                self._func = None
            case FunctionType() as func:
                self._value = func(0)
                self._func = func
            case _:
                raise FactorError('invalid value for Factor, value can be int | float | Callable[[int], float]')

    def get_fvalue(self) -> Callable[[int], float] | float:
        if self._func is not None:
            return self._func
        return self._value

    @staticmethod
    def may_be_factor(value: Any) -> bool:
        if isinstance(value, (int, float)):
            return True
        elif callable(value):
            try:
                result = value(0)
                return isinstance(result, (float, int))
            except Exception as e:
                return False
        else:
            return False

    def update_matrix_value(self):
        if self._connected_matrix is not None and self._value_pos_in_matrix is not None:
            self._connected_matrix[self._value_pos_in_matrix] = self._value

    def update(self, time: int) -> None:
        try:
            res = self._func(time)
        except Exception:
            raise FactorError(f"factor '{self}' cannot be calculated with argument {time}")
        self._value = res
        self._connected_matrix[self._value_pos_in_matrix] = res

    @property
    def value(self) -> float:
        return self._value

    @property
    def is_dynamic(self) -> bool:
        return self._func is not None

    @property
    def name(self) -> str:
        return self._name

    def __str__(self) -> str:
        return self._name

    def set_latex_repr(self, latex_repr: Optional[str]) -> None:
        if latex_repr is not None and not isinstance(latex_repr, str):
            raise FactorError('latex_repr must be str or None')
        self._latex_repr = latex_repr

    def get_latex_repr(self) -> str:
        if self._latex_repr is None:
            return self._name
        return self._latex_repr
