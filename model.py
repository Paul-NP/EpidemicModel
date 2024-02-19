from __future__ import annotations
import csv
from prettytable import PrettyTable

from factor import Factor, factor_one, factor_zero
from stage import Stage
from flow import Flow


class EpidemicModel:
    _stages: list[Stage]
    _flows: list[Flow]
    _factors: list[Factor]
    _simulation_time: int
    _imitation: bool = False
    _model_result: list[list[int | float]]
    _factors_table: list[list[int | float]]

    __len_float: int = 4

    def __init__(self, stages: list[Stage], flows: list[Flow], simulation_time: int,
                 imitation: bool = False):
        """
        EpidemicModel - compartmental epidemic model
        :param stages: list of Stages
        :param flows: list of Flows
        :param simulation_time: num of simulation steps
        :param imitation: flag for imitation or numeric simulation
        """
        if any(not isinstance(st, Stage) for st in stages):
            raise ValueError("all stages in model must be Stage")
        if any(not isinstance(fl, Flow) for fl in flows):
            raise ValueError("all flows in model must be Flow")
        if not isinstance(simulation_time, int):
            raise ValueError("simulation_time must be Int")

        self._stages = stages
        self._flows = flows
        self._fill_factors()
        self._simulation_time = simulation_time
        self.imitation = imitation
        self._model_result = []
        self._factors_table = []

    def _fill_factors(self):
        list_factors = []
        for fl in self._flows:
            if fl.flow_factor not in list_factors:
                list_factors.append(fl.flow_factor)
            if fl.infect_factor not in list_factors:
                list_factors.append(fl.infect_factor)
            for fa in fl.inducing_factors.values():
                if fa not in list_factors:
                    list_factors.append(fa)
        self._factors = list_factors

    @property
    def imitation(self):
        return self._imitation

    @imitation.setter
    def imitation(self, value):
        if not isinstance(value, bool):
            raise ValueError("imitation for Flow must be bool")
        if self._imitation != value:
            for fl in self._flows:
                fl.imitation = value

    def _model_step(self, step: int):
        [fa(step) for fa in self._factors]
        [fl.make_changes() for fl in self._flows]
        [st.apply_changes() for st in self._stages]
        self._model_result.append([st.num for st in self._stages])
        self._factors_table.append([fa.value for fa in self._factors])
        # print(f"#{step}: {self._model_result[-1]}")

    def start(self):
        self._model_result = [[st.num for st in self._stages]]
        for step in range(1, self._simulation_time + 1):
            # print(f"step #{step}")
            self._model_step(step)

    def print_result(self):
        table = PrettyTable()
        table.field_names = ["step"] + [st.name for st in self._stages]
        table.add_rows([[i] + self._model_result[i] for i in range(self._simulation_time)])
        table.float_format = f".{self.__len_float}"
        print(table)

    def print_factors_table(self):
        table = PrettyTable()
        table.field_names = ["step"] + [f.name for f in self._factors]
        table.add_rows([[i] + self._factors_table[i] for i in range(self._simulation_time)])
        table.float_format = f".{self.__len_float}"
        print(table)

    def write_result(self, filename: str, floating_point=".", delimiter=","):
        if not self._model_result:
            raise(ValueError("result list is empty"))
        if not isinstance(filename, str):
            raise(ValueError("filename must be str"))

        with open(filename, "w", encoding="utf-8-sig", newline="") as file:
            csv_writer = csv.writer(file, delimiter=delimiter)
            csv_writer.writerow(["step"] + [st.name for st in self._stages])
            for row_i, row in enumerate(self._model_result):
                csv_writer.writerow([str(row_i)] + [str(value).replace(".", floating_point) for value in row])


# s = Stage("S", 100)
# i = Stage("I", 1)
# r = Stage("R", 0)
# d = Stage("D", 0)
#
# beta = Factor(0.004)
# gamma = Factor(0.1)
# death_rate = Factor(0.2)
#
# si = Flow(s, i, infect_factor=beta, inducing_factors={i: factor_one})
# ir = Flow(i, r, flow_factor=gamma * (1 - death_rate))
# id = Flow(i, d, flow_factor=gamma * death_rate)
#
# model = EpidemicModel([s, i, r], [si, ir], 500, imitation=False)
# model.start()
# model.print_result()
# model.write_result("test_result_1.csv", floating_point=",", delimiter=";")
