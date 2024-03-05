from model import EpidemicModel
from stage import Stage
from flow import Flow
from factor import Factor
import matplotlib.pyplot as plt
import math

def experiment_1():
    s = Stage("S", 100)
    i = Stage("I", 1)
    r = Stage("R", 0)
    d = Stage("D", 0)

    # beta = Factor(Factor.func_by_keyframes({0: 0.004, 50: 0.001}), 'beta')
    # beta = Factor(lambda x: (math.sin(x / 10) + 1) / 40, 'beta')
    # gamma = Factor(0, 'gama')
    beta = 0.004
    gama = 0.1
    delta = 0.01
    # death_rate = 0.1

    si = Flow(s, i, flow_factor=beta, inducing={i: 1})
    ir = Flow(i, r, flow_factor=gama)
    # ird = Flow(i, {r: 1 - death_rate, d: death_rate}, flow_factor=gamma)
    rs = Flow(r, s, flow_factor=delta)

    model = EpidemicModel([s, i, r], [si, rs, ir], 200)
    model.start()

    model.print_result_table()
    model.print_flows_table()
    model.print_factors_table()

    model.write_all_result('SIRS_1')

    # model.factors_df.plot()
    model.flows_df.plot()
    model.result_df.plot()
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
    experiment_1()
