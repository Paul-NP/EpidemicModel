from epidemmo.model import EpidemicModel
from epidemmo.builder import ModelBuilder
import pytest

from epidemmo.standard import Standard


@pytest.fixture()
def seirds_latex_full_relative() -> str:
    return '\\begin{equation}\\label{eq:SEIRDS_full}\n    \\begin{cases}\n        \\frac{dS}{dt} = R \\cdot \\sigma - S \\cdot (1 - (1 - \\frac{\\beta}{N})^{I})\\\\\n        \\frac{dE}{dt} = S \\cdot (1 - (1 - \\frac{\\beta}{N})^{I}) - E \\cdot \\alpha\\\\\n        \\frac{dI}{dt} = E \\cdot \\alpha - I \\cdot \\gamma\\\\\n        \\frac{dR}{dt} = I \\cdot \\gamma \\cdot (1-\\delta) - R \\cdot \\sigma\\\\\n        \\frac{dD}{dt} = I \\cdot \\gamma \\cdot \\delta\\\\\n    \\end{cases}\n\\end{equation}\n'

@pytest.fixture()
def sir_result10() -> list[float]:
    # при beta = 0.4, gamma = 0.1
    # 10 шагов симуляции, в одном списке сначала результаты по S, потом по I, потом по R
    result = [99.0, 98.6, 98.09, 97.44, 96.59, 95.52, 94.15, 92.44, 90.29, 87.65, 1.0, 1.3, 1.68, 2.17, 2.79, 3.59, 4.59, 5.85, 7.41, 9.31, 0.0, 0.1, 0.23, 0.4, 0.61, 0.89, 1.25, 1.71, 2.3, 3.04]
    return result

@pytest.fixture()
def sir_result10_beta05_gamma02() -> list[float]:
    # при beta = 0.5, gamma = 0.2
    result = [99.0, 98.5, 97.87, 97.05, 96.01, 94.68, 93.02, 90.95, 88.4, 85.32, 1.0, 1.3, 1.67, 2.16, 2.77, 3.54, 4.49, 5.67, 7.08, 8.75, 0.0, 0.2, 0.46, 0.79, 1.22, 1.78, 2.49, 3.38, 4.52, 5.93]
    return result

@pytest.fixture()
def seir_result10() -> list[float]:
    # при beta = 0.4, gamma = 0.1, alpha = 0.1
    result = [99.0, 98.6, 98.25, 97.91, 97.59, 97.26, 96.91, 96.55, 96.16, 95.74, 0.0, 0.4, 0.71, 0.97, 1.2, 1.42, 1.62, 1.82, 2.03, 2.24, 1.0, 0.9, 0.85, 0.84, 0.85, 0.89, 0.94, 1.01, 1.09, 1.18, 0.0, 0.1, 0.19, 0.27, 0.36, 0.44, 0.53, 0.63, 0.73, 0.84]
    return result

@pytest.fixture()
def slhrd_result10() -> list[float]:
    # при beta = 0.4, gamma = 0.1, смертности = 0.4, вероятность тяжёлой формы = 0.2
    result = [99.0, 98.6, 98.09, 97.44, 96.59, 95.52, 94.15, 92.44, 90.29, 87.65, 1.0, 1.22, 1.5, 1.88, 2.37, 2.99, 3.78, 4.78, 6.01, 7.53, 0.0, 0.08, 0.17, 0.29, 0.43, 0.6, 0.81, 1.07, 1.4, 1.78, 0.0, 0.1, 0.23, 0.39, 0.59, 0.85, 1.19, 1.62, 2.16, 2.84, 0.0, 0.0, 0.0, 0.01, 0.02, 0.04, 0.06, 0.1, 0.14, 0.19]
    return result


def test_sir(sir_result10):
    builder = ModelBuilder()
    builder.add_stage('S', 99).add_stage('I', 1).add_stage('R')
    builder.add_factor('beta', 0.4).add_factor('gamma', 0.1)
    builder.add_flow('S', 'I', 'beta', 'I').add_flow('I', 'R', 'gamma')

    model = builder.build()

    result = model.start(10).to_numpy().round(2).T.ravel().tolist()
    assert result== pytest.approx(sir_result10, abs=0.01)


def test_seir(seir_result10):
    builder = ModelBuilder()
    builder.add_stage('S', 99).add_stage('E', 0).add_stage('I', 1).add_stage('R')
    builder.add_factor('beta', 0.4).add_factor('gamma', 0.1).add_factor('alpha', 0.1)
    builder.add_flow('S', 'E', 'beta', 'I').add_flow('E', 'I', 'alpha').add_flow('I', 'R', 'gamma')

    model = builder.build()

    result = model.start(10).to_numpy().round(2).T.ravel().tolist()
    assert result == pytest.approx(seir_result10, abs=0.01)


def test_sir_stoch():
    model = Standard.get_SIR_builder().build()
    result = model.start(50, stochastic=True)
    assert result.sum(axis=1).to_list() == pytest.approx([100] * 50, abs=0.01)


def test_slhrd(slhrd_result10):
    builder = ModelBuilder()
    builder.add_stage('S', 99).add_stage('L', 1).add_stage('H', 0).add_stage('R').add_stage('D')
    builder.add_factor('beta', 0.4).add_factor('gamma', 0.1)
    builder.add_flow('S', {'L': 0.8, 'H': 0.2}, 'beta', {'L': 1, 'H': 1})
    builder.add_flow('H', {'R': 0.6, 'D': 0.4}, 'gamma')
    builder.add_flow('L', 'R', 'gamma')

    model = builder.build()
    result = model.start(10).to_numpy().round(2).T.ravel().tolist()
    assert result == pytest.approx(slhrd_result10, abs=0.01)


def test_sir_fast_create(sir_result10):
    builder = ModelBuilder()
    builder.add_stages(S=99, I=1, R=0).add_factors(beta=0.4, gamma=0.1)
    builder.add_flow('S', 'I', 'beta', 'I').add_flow('I', 'R', 'gamma')

    model = builder.build()
    result = model.start(10).to_numpy().round(2).T.ravel().tolist()
    assert result == pytest.approx(sir_result10, abs=0.01)


def test_standard_sir(sir_result10):
    model = Standard.get_SIR_builder().build()
    result = model.start(10)
    result = model.start(10).to_numpy().round(2).T.ravel().tolist()
    assert result == pytest.approx(sir_result10, abs=0.01)


def test_standard_seir(seir_result10):
    model = Standard.get_SEIR_builder().build()
    result = model.start(10)
    result = model.start(10).to_numpy().round(2).T.ravel().tolist()
    assert result == pytest.approx(seir_result10, abs=0.01)


def test_sir_changed(sir_result10_beta05_gamma02):
    model = Standard.get_SIR_builder().build()
    model.set_factors(beta=0.5, gamma=0.2)
    result = model.start(10).to_numpy().round(2).T.ravel().tolist()
    assert result == pytest.approx(sir_result10_beta05_gamma02, abs=0.01)


def test_seirds_latex(seirds_latex_full_relative):
    model = Standard.get_SEIRDS_builder().build()
    latex = model.get_latex()
    assert latex == seirds_latex_full_relative


def test_dynamic_sir(sir_result10):
    model = Standard.get_SIR_builder().build()
    model.set_factors(beta=lambda x: 0.4, gamma=[0.1]*20)
    result = model.start(10).to_numpy().round(2).T.ravel().tolist()
    assert result == pytest.approx(sir_result10, abs=0.01)