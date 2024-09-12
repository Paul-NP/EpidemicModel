from dataclasses import dataclass
from typing import Callable, Literal, Union, Type, TypeAlias
from .model import EpidemicModel
from .builder import ModelBuilder, ModelBuilderError

import json


factorValue: TypeAlias = Union[int, float, Callable[[int], float]]


class ModelIOError(Exception):
    pass


class AbstractIO:
    def load(self, source: str) -> EpidemicModel:
        try:
            return self._parse(source)
        except ModelBuilderError as e:
            e.add_note('(while json parser work)')
            raise e
        except Exception as e:
            raise ModelIOError(f'While jsonIO loading model: {str(e)}')

    def dump(self, model: EpidemicModel) -> str:
        try:
            return self._generate(model)
        except Exception as e:
            raise ModelIOError(f'While jsonIO dumping model: {str(e)}')

    def _parse(self, source: str) -> EpidemicModel:
        raise ModelIOError(f'{type(self)} does not support file parsing')

    def _generate(self, model: EpidemicModel) -> str:
        raise ModelIOError(f'{type(self)} does not support file generation')


class KK2024(AbstractIO):
    def _parse(self, source: str) -> EpidemicModel:
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


class SimpleModel(AbstractIO):
    def _parse(self, source: str) -> EpidemicModel:
        structure = json.loads(source)

        builder = ModelBuilder()
        builder.set_model_name(str(structure['model_name']))

        stages = {str(st['name']): float(st['num']) for st in structure['stages']}
        builder.add_stages(**stages)

        factors = {str(fa['name']): float(fa['value']) for fa in structure['factors']}
        builder.add_factors(**factors)

        for fl in structure['flows']:
            builder.add_flow(**fl)

        return builder.build()

    def _generate(self, model: EpidemicModel) -> str:
        structure = {'stages': model.stages, 'factors': model.factors, 'flows': model.flows}
        return json.dumps(structure, indent=4)


class ModelIO:
    io_ways: dict[str, Type[AbstractIO]] = {'kk_2024': KK2024, 'simple': SimpleModel}

    def __init__(self, struct_version: Literal['kk_2024'] = 'simple') -> None:
        if struct_version not in self.io_ways:
            raise ModelIOError('Unknown structure version')

        self._io: AbstractIO = self.io_ways[struct_version]()

    def load(self, filename: str) -> EpidemicModel:
        with open(filename, 'r', encoding='utf8') as file:
            json_string = file.read()
            return self._io.load(json_string)

    def save(self, model: EpidemicModel, filename: str) -> None:
        json_string = self._io.dump(model)
        with open(filename, 'w', encoding='utf8') as file:
            file.write(json_string)
