from __future__ import annotations


class Stage:
    _name: str
    _current_num: float
    _changes: list[float | int]

    __name_len: int = 3
    __num_len: int = 4

    def __init__(self, name: str, start_num: float = 0):
        if not isinstance(name, str):
            raise ValueError("Stage name must be str")
        if not isinstance(start_num, float | int):
            raise ValueError("Stage start num must be number")

        self._name = name
        self._current_num = float(start_num)
        self._changes = []

    def add_change(self, change: float | int):
        if not isinstance(change, float | int):
            raise ValueError("Stage change must be number")
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

    def __str__(self) -> str:
        return f"Stage '{self._name}'"

    def __repr__(self) -> str:
        return f"Stage '{self._name:^{self.__name_len}s}' | '{self._current_num:.{self.__num_len}f}'"
