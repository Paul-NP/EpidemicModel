import pytest

import numpy as np
from epidemmo.standard import Standard
from epidemmo.fit_model import ModelFitter

@pytest.mark.parametrize('beta, gamma', [(0.5, 0.1), (0.2, 0.05), (0.9, 0.1)])
def test_fitter_only_factors(beta, gamma):
    np.random.seed(0)

    model = Standard.get_SIR_builder().build()
    model.set_start_stages(S=99, I=1, R=0)
    model.set_factors(beta=beta, gamma=gamma)

    model.start(60, full_save=True)
    real_data = model.flows_df[['Flow(S>I)', 'Flow(I>R)']]

    new_model = Standard.get_SIR_builder().build()
    new_model.set_start_stages(S=99, I=1, R=0)
    new_model.set_factors(beta=0, gamma=0)
    fitter = ModelFitter(new_model)
    fitter.set_changeable_factors({'beta': (0, 1), 'gamma': (0, 1)})
    fitter.fit(real_data)

    beta_pr, gamma_pr = new_model.factors_dict['beta'], new_model.factors_dict['gamma']
    print(beta, beta_pr, gamma, gamma_pr)
    assert beta_pr == pytest.approx(beta, abs=0.1) and gamma_pr == pytest.approx(gamma, abs=0.1)


@pytest.mark.parametrize('beta, gamma, delta', [(0.5, 0.1, 3), (0.2, 0.05, 5), (0.9, 0.1, 6)])
def test_fitter_only_factors_with_interval(beta, gamma, delta):
    np.random.seed(0)

    model = Standard.get_SIR_builder().build()
    model.set_start_stages(S=999, I=1, R=0)
    model.set_factors(beta=beta, gamma=gamma)

    model.start(int(60 / delta), delta=delta, full_save=True)
    real_data = model.flows_df[['Flow(S>I)', 'Flow(I>R)']]

    new_model = Standard.get_SIR_builder().build()
    new_model.set_start_stages(S=999, I=1, R=0)
    new_model.set_factors(beta=0, gamma=0)
    fitter = ModelFitter(new_model)
    fitter.set_changeable_factors({'beta': (0, 1), 'gamma': (0, 1)})
    fitter.fit(real_data, interval=delta)

    beta_pr, gamma_pr = new_model.factors_dict['beta'], new_model.factors_dict['gamma']
    print(beta, beta_pr, gamma, gamma_pr)
    assert beta_pr == pytest.approx(beta, abs=0.1) and gamma_pr == pytest.approx(gamma, abs=0.1)