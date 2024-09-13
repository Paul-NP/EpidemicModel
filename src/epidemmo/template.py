from .builder import ModelBuilder


class Standard:
    @staticmethod
    def SIR_builder() -> ModelBuilder:
        builder = ModelBuilder().add_stages(S=100, I=1, R=0).add_factors(beta=0.4, gamma=0.1)
        builder.add_flow('S', 'I', 'beta', 'I').add_flow('I', 'R', 'gamma')
        builder.set_model_name('SIR')
        return builder

    @staticmethod
    def SEIR_builder() -> ModelBuilder:
        builder = ModelBuilder().add_stages(S=100, E=0, I=1, R=0).add_factors(beta=0.4, gamma=0.1, alpha=0.1)
        builder.add_flow('S', 'E', 'beta', 'I').add_flow('E', 'I', 'alpha').add_flow('I', 'R', 'gamma')
        builder.set_model_name('SEIR')
        return builder

    @staticmethod
    def SIRD_builder() -> ModelBuilder:
        builder = ModelBuilder().add_stages(S=100, I=1, R=0, D=0).add_factors(beta=0.4, gamma=0.1)
        builder.add_flow('S', 'I', 'beta', 'I').add_flow('I', {'R': 0.8, 'D': 0.2}, 'gamma')
        builder.set_model_name('SIRD')
        return builder

    @staticmethod
    def SIRS_builder() -> ModelBuilder:
        builder = ModelBuilder().add_stages(S=100, I=1, R=0).add_factors(beta=0.5, gamma=0.2, delta=0.01)
        builder.add_flow('S', 'I', 'beta', 'I').add_flow('I', 'R', 'gamma').add_flow('R', 'S', 'delta')
        builder.set_model_name('SIRS')
        return builder
