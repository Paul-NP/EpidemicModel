from __future__ import annotations


class StageError(Exception):
    pass


class Stage:
    __MIN_NAME_LEN: int = 1
    __MAX_NAME_LEN: int = 4
    __FLOAT_LEN: int = 2

    def __init__(self, name: str, start_num: float = 0):
        if not isinstance(name, str):
            raise StageError("Stage name must be str")
        if not self.__MIN_NAME_LEN <= len(name) <= self.__MAX_NAME_LEN:
            raise StageError("Stage name have unexpected len")
        if not isinstance(start_num, float | int):
            raise StageError("Stage start num must be number")

        self._name: str = name
        self._current_num: float = float(start_num)
        self._changes: list[float | int] = []

    def add_change(self, change: float | int):
        if not isinstance(change, float | int):
            raise StageError("Stage change must be number")
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
        return f"Stage <{self._name}>"

    def __repr__(self) -> str:
        return f"Stage <{self._name}> | {self._current_num:.{self.__FLOAT_LEN}f}"
