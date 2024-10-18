from dataclasses import dataclass
from typing import Callable, Literal, Union, Type, TypeAlias
from .model import EpidemicModel
from .fast_model import FastEpidemicModel
from .builder import ModelBuilder, ModelBuilderError
from .fast_builder import FastModelBuilder, FastModelBuilderError

import json


factorValue: TypeAlias = Union[int, float, Callable[[int], float]]


class FastModelIOError(Exception):
    pass


class FastAbstractIO:
    def load(self, source: str) -> FastEpidemicModel:
        try:
            return self._parse(source)
        except ModelBuilderError as e:
            e.add_note('(while json parser work)')
            raise e
        except Exception as e:
            raise FastModelIOError(f'While jsonIO loading model: {str(e)}')

    def dump(self, model: FastEpidemicModel) -> str:
        try:
            return self._generate(model)
        except Exception as e:
            raise FastModelIOError(f'While jsonIO dumping model: {str(e)}')

    def _parse(self, source: str) -> FastEpidemicModel:
        raise FastModelIOError(f'{type(self)} does not support file parsing')

    def _generate(self, model: FastEpidemicModel) -> str:
        raise FastModelIOError(f'{type(self)} does not support file generation')


class FastKK2024IO(FastAbstractIO):
    def _parse(self, source: str) -> FastEpidemicModel:
        structure = json.loads(source)
        raw_stages = structure['compartments']
        raw_flows = structure['flows']

        builder = ModelBuilder()

        stages = {st['name']: st['population'] for st in raw_stages}
        builder.add_stages(**stages)

        for r_flow in raw_flows:
            start = str(r_flow['from'])
            end_dict: dict[str, str | factorValue] = {str(end['name']): float(end['coef']) for end in r_flow['to']}
            ind_dict: dict[str, str | factorValue] = {}
            if 'induction' in r_flow:
                ind_dict = {str(ind['name']): float(ind['coef']) for ind in r_flow['induction']}

            fl_factor = float(r_flow['coef'])
            builder.add_flow(start, end_dict, fl_factor, ind_dict)

        return builder.build()


class FastSimpleIO(FastAbstractIO):
    def _parse(self, source: str) -> FastEpidemicModel:
        structure = json.loads(source)

        builder = ModelBuilder()
        builder.set_model_name(str(structure['name']))

        stages = {str(st['name']): float(st['num']) for st in structure['stages']}
        builder.add_stages(**stages)

        factors = {str(fa['name']): float(fa['value']) for fa in structure['factors']}
        builder.add_factors(**factors)

        for fl in structure['flows']:
            builder.add_flow(**fl)

        return builder.build()

    def _generate(self, model: FastEpidemicModel) -> str:
        structure = {'name': model.name, 'stages': model.stages, 'factors': model.factors, 'flows': model.flows}
        return json.dumps(structure, indent=4)


class FastModelIO:
    io_ways: dict[str, Type[FastAbstractIO]] = {'kk_2024': FastKK2024IO, 'simple_io': FastSimpleIO}

    def __init__(self, struct_version: Literal['kk_2024', 'simple_io'] = 'simple_io') -> None:

        if struct_version not in self.io_ways:
            raise FastModelIOError('Unknown structure version')

        self._io: FastAbstractIO = self.io_ways[struct_version]()

    def load(self, filename: str) -> FastEpidemicModel:
        with open(filename, 'r', encoding='utf8') as file:
            json_string = file.read()
            return self._io.load(json_string)

    def save(self, model: FastEpidemicModel, filename: str) -> None:
        json_string = self._io.dump(model)
        with open(filename, 'w', encoding='utf8') as file:
            file.write(json_string)
