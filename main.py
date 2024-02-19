from model import EpidemicModel
from factor import Factor
from stage import Stage
from flow import Flow
from show_graphs import show_result


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

sm = Flow(s, m, infect_factor=beta * beta_m, flow_factor=one, inducing_factors={m: one, sa: one})
ssa = Flow(s, sa, infect_factor=beta * beta_sa, flow_factor=one, inducing_factors={m: one, sa: one})
mr = Flow(m, r, flow_factor=r_mild)
md = Flow(m, d, flow_factor=d_mild)
sar = Flow(sa, r, flow_factor=r_sa)
sad = Flow(sa, d, flow_factor=d_sa)
hr = Flow(h, r, flow_factor=r_help)
hd = Flow(h, d, flow_factor=d_help)
sah = Flow(sa, h, flow_factor=sah_f)


model = EpidemicModel(stages=[s, m, sa, r, d, h], flows=[sm, ssa, mr, md, sar, sad, hr, hd, sah], simulation_time=500)
model.start()
model.print_result()
model.write_result('result.csv')
show_result('temp/result.csv', delimiter=',')



# r = Stage("R", 0)
# d = Stage("D", 0)
#
# beta = Factor(0.004)
# gamma = Factor(0.1)
# death_rate = Factor(0.2)
#
# si = Flow(s, i, infect_factor=beta, inducing_factors={i: factor_one})
# ir = Flow(i, r, flow_factor=gamma * (1 - death_rate))
# id = Flow(i, d, flow_factor=gamma * death_rate)
#
# model = EpidemicModel([s, i, r], [si, ir], 500, imitation=False)
# model.start()
# model.print_result()