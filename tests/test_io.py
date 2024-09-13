import pytest
from epidemmo import ModelIO, Standard


def test_save_sir(tmp_path):
    filename = str(tmp_path / 'sir_model.json')
    io = ModelIO()
    sir = Standard.get_SIR_builder().build()
    io.save(sir, filename)
    sir2 = io.load(filename)
    assert repr(sir2) == 'Model(SIR): [F(S>I|by-I), F(I>R|by-)]'

