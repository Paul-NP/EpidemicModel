from .builder import ModelBuilder


class Standard:
    @staticmethod
    def get_SIR_builder() -> ModelBuilder:
        builder = ModelBuilder().add_stages(S=99, I=1, R=0)
        builder.add_factor('beta', 0.4, latex_repr=r'\beta').add_factor('gamma', 0.1, latex_repr=r'\gamma')
        builder.add_flow('S', 'I', 'beta', 'I').add_flow('I', 'R', 'gamma')
        builder.set_model_name('SIR')
        return builder

    @staticmethod
    def get_SEIR_builder() -> ModelBuilder:
        builder = ModelBuilder().add_stages(S=99, E=0, I=1, R=0)
        builder.add_factor('beta', 0.4, latex_repr=r'\beta').add_factor('gamma', 0.1, latex_repr=r'\gamma')
        builder.add_factor('alpha', 0.1, latex_repr=r'\alpha')
        builder.add_flow('S', 'E', 'beta', 'I').add_flow('E', 'I', 'alpha').add_flow('I', 'R', 'gamma')
        builder.set_model_name('SEIR')
        return builder

    @staticmethod
    def get_SIRD_builder() -> ModelBuilder:
        builder = ModelBuilder().add_stages(S=99, I=1, R=0, D=0)
        builder.add_factor('beta', 0.4, latex_repr=r'\beta').add_factor('gamma', 0.1, latex_repr=r'\gamma')
        builder.add_factor('delta', 0.2, latex_repr=r'\delta').add_factor('no_delta', 0.8, latex_repr=r'(1-\delta)')
        builder.add_flow('S', 'I', 'beta', 'I').add_flow('I', {'R': 'no_delta', 'D': 'delta'}, 'gamma')
        builder.set_model_name('SIRD')
        return builder

    @staticmethod
    def get_SIRS_builder() -> ModelBuilder:
        builder = ModelBuilder().add_stages(S=99, I=1, R=0)
        builder.add_factor('beta', 0.5, latex_repr=r'\beta').add_factor('gamma', 0.2, latex_repr=r'\gamma')
        builder.add_factor('sigma', 0.01, latex_repr=r'\sigma')
        builder.add_flow('S', 'I', 'beta', 'I').add_flow('I', 'R', 'gamma').add_flow('R', 'S', 'sigma')
        builder.set_model_name('SIRS')
        return builder

    @staticmethod
    def get_SEIRDS_builder() -> ModelBuilder:
        builder = ModelBuilder().add_stages(S=99, E=0, I=1, R=0, D=0)
        builder.add_factor('beta', 0.5, latex_repr=r'\beta').add_factor('gamma', 0.2, latex_repr=r'\gamma')
        builder.add_factor('delta', 0.2, latex_repr=r'\delta').add_factor('no_delta', 0.8, latex_repr=r'(1-\delta)')
        builder.add_factor('alpha', 0.1, latex_repr=r'\alpha').add_factor('sigma', 0.01, latex_repr=r'\sigma')

        builder.add_flow('S', 'E', 'beta', 'I').add_flow('E', 'I', 'alpha')
        builder.add_flow('I', {'R': 'no_delta', 'D': 'delta'}, 'gamma').add_flow('R', 'S', 'sigma')

        builder.set_model_name('SEIRDS')
        return builder




