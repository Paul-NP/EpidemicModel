from typing import Callable, Literal, Union, Type, TypeAlias
import abc
from .model import EpidemicModel
from .builder import ModelBuilder, ModelBuilderError

import json


factorValue: TypeAlias = Union[int, float, Callable[[int], float]]


class ModelIOError(Exception):
    pass


class VersionIO(abc.ABC):
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

    @abc.abstractmethod
    def _parse(self, source: str) -> EpidemicModel:
        pass

    @abc.abstractmethod
    def _generate(self, model: EpidemicModel) -> str:
        pass


class KK2024(VersionIO):
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

    def _generate(self, model: EpidemicModel) -> str:
        raise ModelIOError(f'kk_2024 does not support file generation')


class ModelIO:
    io_ways: dict[str, Type[VersionIO]] = {'kk_2024': KK2024}

    def __init__(self, struct_version: Literal['kk_2024']) -> None:
        if struct_version not in self.io_ways:
            raise ModelIOError('Unknown structure version')

        self._io: VersionIO = self.io_ways[struct_version]()

    def from_json(self, filename: str) -> EpidemicModel:
        with open(filename, 'r', encoding='utf8') as file:
            json_string = file.read()
            return self._io.load(json_string)

    def to_json(self, model: EpidemicModel, filename: str) -> None:
        json_string = self._io.dump(model)
        with open(filename, 'w', encoding='utf8') as file:
            file.write(json_string)

