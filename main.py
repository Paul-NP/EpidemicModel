from model import EpidemicModel
from stage import Stage
from flow import Flow
from factor import Factor
import matplotlib.pyplot as plt
import math


def sirs():
    s = Stage("S", 100)
    i = Stage("I", 1)
    r = Stage("R", 0)

    beta = 0.004
    gama = 0.1
    delta = 0.01

    si = Flow(s, i, flow_factor=beta, inducing={i: 1})
    ir = Flow(i, r, flow_factor=gama)
    rs = Flow(r, s, flow_factor=delta)

    model = EpidemicModel([s, i, r], [si, rs, ir])
    model.start(300)

    model.print_result_table()
    model.print_flows_table()
    model.print_factors_table()

    model.write_all_result('SIRS_1')

    model.flows_df.plot()
    model.result_df.plot()
    plt.show()


def sir():
    s = Stage('S', 1000)
    i = Stage('I', 1)
    r = Stage('R')

    beta = Factor(0.004, name='beta')
    gama = Factor(0.1, name='gama')

    si = Flow(s, i, beta, inducing=i)
    ir = Flow(i, r, gama)

    model = EpidemicModel((s, i, r), (si, ir))

    for i in range(10):
        result = model.start(500, stochastic_time=True, stochastic_changes=True, beta=0.00002, gama=0.015)
        plt.plot(result['I'], label=f'result_{i}')
    result_teor = model.start(500, beta=0.00002, gama=0.015)
    plt.plot(result_teor['I'], label='result_teor')
    # model.print_full_result()
    # model.result_df.plot()
    # model.print_result_table()
    # model.print_flows_table()
    # model.print_factors_table()
    # result1.plot()

    # result2 = model.start(200, beta=0.001, gama=0.13)
    # result3 = model.start(200, beta=0.001, gama=0.16)



    # plt.plot(result3['I'], label='result3')
    # model.print_result_table()
    plt.legend()
    plt.show()


def experiment_2():
    s = Stage("S", 100000)
    m = Stage("M", 1)
    sa = Stage("Se", 0)
    r = Stage("R", 0)
    d = Stage("D", 0)
    h = Stage("H", 0)

    one = Factor(1)
    beta = Factor(0.2 / s.num)
    beta_m = Factor(0.7)
    beta_sa = Factor(0.3)
    r_mild = Factor(0.075)
    d_mild = Factor(0.025)
    r_sa = Factor(0.05)
    d_sa = Factor(0.05)
    r_help = Factor(0.09)
    d_help = Factor(0.01)
    num_bet = 200 * 0.95
    soglas = 0.9

    sah_f = Factor(lambda x: soglas * min(1.0, (num_bet - h.num) / sa.num if sa.num else 1.0))
    # sah_f = Factor(1)

    sm = Flow(s, m, infect_factor=beta * beta_m, flow_factor=one, inducing={m: one, sa: one})
    ssa = Flow(s, sa, infect_factor=beta * beta_sa, flow_factor=one, inducing={m: one, sa: one})
    mr = Flow(m, r, flow_factor=r_mild)
    md = Flow(m, d, flow_factor=d_mild)
    sar = Flow(sa, r, flow_factor=r_sa)
    sad = Flow(sa, d, flow_factor=d_sa)
    hr = Flow(h, r, flow_factor=r_help)
    hd = Flow(h, d, flow_factor=d_help)
    sah = Flow(sa, h, flow_factor=sah_f)

    model = EpidemicModel(stages=[s, m, sa, r, d, h], flows=[sm, ssa, mr, md, sar, sad, hr, hd, sah],
                          simulation_time=500)
    model.start()
    model.print_result_table()
    model.write_result('result.csv')
    show_result('temp/result.csv', delimiter=',')


if __name__ == '__main__':
    sir()
