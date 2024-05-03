import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

from model import EpidemicModel
from stage import Stage
from flow import Flow
from factor import Factor
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize
import scipy.stats as sts


def sir_exp():
    s = Stage('S', 1000)
    i = Stage('I', 10)
    r = Stage('R')

    beta = Factor(0.0002, name='beta')
    gama = Factor(0.05, name='gama')

    si = Flow(s, i, beta, inducing_factors=i)
    ir = Flow(i, r, gama)

    model = EpidemicModel((s, i, r), (si, ir))

    # result_teor = model.start(500, beta=0.0002, gama=0.01)
    # model.print_full_result()
    # model.result_df.plot()
    # model.print_result_table()
    # model.print_flows_table()
    # model.print_factors_table()
    # result1.plot()
    # result1 = model.start(200, beta=0.001, gama=0.1)
    # result2 = model.start(200, beta=0.001, gama=0.13)

    result = model.start(200)
    plt.plot(result['S'], label='teor')
    n = 30
    stoch_result_min = model.start(200, stochastic_changes=True)['S']
    stoch_result_max = stoch_result_min.copy()

    for i in range(n):
        print(f'stoch #{i+1}')

        result = model.start(200, stochastic_changes=True)['S']

        # print(stoch_result_min)
        # print(stoch_result_max)
        # print(result)

        stoch_result_min = np.array([stoch_result_min, result]).min(axis=0)
        stoch_result_max = np.array([stoch_result_max, result]).max(axis=0)

        # plt.plot(model.start(200, stochastic_changes=True, stochastic_time=True)['I'], label=f'shoch #{i+1}')
    plt.plot(stoch_result_min, label='min')
    plt.plot(stoch_result_max, label='max')
    # plt.plot(result1['I'], label='result1')
    # plt.plot(result2['I'], label='result2')

    # model.print_result_table()
    plt.legend()
    plt.show()

def sir_exp2():
    s = Stage('S', 1000)
    i = Stage('I', 10)
    r = Stage('R')

    beta = Factor(0.0001, name='beta')
    gama = Factor(0.05, name='gama')

    si = Flow(s, i, beta, inducing_factors=i)
    ir = Flow(i, r, gama)

    model = EpidemicModel((s, i, r), (si, ir))

    # params = np.array([2.07060536e-05, 6.89766919e-02, 2.50552352e+04])
    # model.set_factors(beta=params[0], gama=params[1])
    # model.set_start_stages(S=params[2])

    model.start(200)['I'].plot()
    # model.flows_df['F(S>I)'].plot()

    for i in range(20):
        model.start(200, stochastic_time=True, stochastic_changes=True)['I'].plot(alpha=0.1)

    plt.show()

    # model.start(200, stochastic_time=True, stochastic_changes=True)
    # model.print_full_result()


def gen_data():
    # np.random.seed(2)
    s = Stage('S', 1000)
    i = Stage('I', 10)
    r = Stage('R')

    beta = Factor(0.0001, name='beta')
    gama = Factor(0.05, name='gama')

    si = Flow(s, i, beta, inducing_factors=i)
    ir = Flow(i, r, gama)

    model = EpidemicModel((s, i, r), (si, ir))

    res = model.start(200, stochastic_changes=True)
    # print(model.full_df[:30])
    model.flows_df.iloc[:, 0].to_csv('gen_data.csv', index=False)


def direct_sir(beta, gama, time):
    result = [[1000, 1, 0]]
    n = sum(result[0])
    for i in range(time):
        si = beta * result[-1][0] * result[-1][1] / n
        ir = gama * result[-1][1]
        result_new = [result[-1][0] - si, result[-1][1] + si - ir, result[-1][2] + ir]
        result.append(result_new)
    return pd.DataFrame(result, columns=['S', 'I', 'R'])


def direct_estimate(data: pd.DataFrame):
    print('start esimate')
    s, i, r = data['S'].array, data['I'].array, data['R'].array
    n = len(data['S'])
    N = s[0] + i[0] + r[0]
    beta = ((-4 / N * (i[:-2] * i[1:-1] * s[1:-1] - i[1:-1] * s[:-2] * s[1: -1] -
                       i[1:-1]*i[2:]*s[1:-1] + i[1:-1] * s[2:] * s[1:-1]).sum() * (i[1:-1]**2).sum() +
            2 / N * ((i[1:-1] * r[2:] - i[1:-1] * r[:-2]).sum() - i[0] * i[1] + i[-2] * i[-1]) * (i[1:-1]**2 * s[1:-1]).sum()) /
            (16 / N**2 * (i[1:-1]**2 * s[1:-1]**2).sum() * (i[1:-1]**2).sum() - 4 / N**2 * (i[1:-1]**2 * s[1:-1]).sum()**2))

    gama = ((4 / N**2 * (i[1:-1]**2 * s[1:-1]**2).sum() * ((i[1:-1]*r[2:] - i[1:-1]*r[:-2]).sum() - i[0]*i[1] - i[-2]*i[-1])
            -2 / N**2 * beta * (i[1:-1]**2 * s[1:-1]).sum() * (i[:-2] * i[1:-1] * s[1:-1] - i[1:-1]*s[:-2]*s[1:-1] - i[1:-1]*i[:-2]*s[1:-1]+i[1:-1]*s[2:]*s[1:-1]).sum()) /
            (16 / N**2 * (i[1:-1]**2 * s[1:-1]**2).sum() * (i[1:-1]**2).sum() - 4/N**2 * (i[1:-1]**2 * s[1:-1]).sum()**2))
    print(beta)
    print(gama)


class PredictModel:
    def __init__(self, filename: str, i, train_time: int = None):
        self.data = pd.read_csv(filename, index_col=0)
        self.data.index = pd.Series(pd.to_datetime(self.data.index)).dt.strftime('%d.%m.%y')
        if train_time is None:
            train_time = len(self.data)
        print(f'{train_time=}')
        # train_time = int(len(self.data) * train_time)
        self.train_data = self.data[:train_time]
        self.n = train_time - 1
        self.sir = self.get_sir(i)
        self.learning_predict = []

    def get_sir(self, i_num):
        self.s = Stage('S', 0)
        self.i = Stage('I', i_num)
        self.r = Stage('R', 0)

        beta = Factor(0.0001, name='beta')
        gama = Factor(0.05, name='gama')

        si = Flow(self.s, self.i, beta, inducing_factors=self.i)
        ir = Flow(self.i, self.r, gama)

        model = EpidemicModel((self.s, self.i, self.r), (si, ir))
        return model

    def predict(self, beta, gama, s):
        self.s.num = s
        self.sir.start(self.n, beta=beta, gama=gama)
        return self.sir.flows_df['F(S>I)']

    def get_error(self, bgs):
        predict = self.predict(*bgs)
        mse = mean_squared_error(y_true=self.train_data, y_pred=predict)
        self.learning_predict.append(predict)
        print(f'{mse: <20} - {bgs}')
        # plt.plot(self.data)
        # plt.plot(predict)
        # plt.show()
        return mse

    def predict_show(self, bgs, n, ax):
        beta, gama, s = bgs
        self.s.num = s
        self.sir.start(n, beta=beta, gama=gama)
        predict = self.sir.flows_df['F(S>I)']
        ax.plot(self.data, label='real_data')
        ax.plot(predict, label='predict')
        # ax.axvline(len(self.train_data))
        # ax.legend()
        # ax.show()


def estimate_params():
    fig, ax = plt.subplots()
    ffamily = 'Cambria'
    fsize = 14
    predict_model = PredictModel('temp/inf_eu_24_new.csv', 10)
    result = minimize(predict_model.get_error, np.array([0.000004, 0.1, 100000]),
                      bounds=[(0, 1), (0, 1), (10_000, 10_000_000)], method='Nelder-Mead')

    ax.set_title('обучение модели на реальных данных', family=ffamily, fontsize=fsize)
    ax.set_ylabel('новых случаев заражения', family=ffamily, fontsize=fsize)

    plt.xticks(rotation=70)
    ax.plot(predict_model.data.index.values, predict_model.data, label='реальные данные', color='red')
    plt.savefig('images_intervals/final_estimate.png', dpi=300)

    line, = ax.plot(predict_model.learning_predict[0][:-1], color='grey', alpha=0.7, label='модель')
    # ax.set_xlabel('время (у.е.)', family=ffamily, fontsize=fsize)
    def anim_func(x):
        line.set_data(np.arange(len(predict_model.learning_predict[x][:-1])),
                      predict_model.learning_predict[x][:-1])
        return [line]

    interv = 60 // 5
    anim = FuncAnimation(
        fig,
        func=anim_func, frames=len(predict_model.learning_predict),
        interval=interv,
        blit=True,
        repeat=False
    )
    plt.legend(loc='upper left')

    writer = PillowWriter(fps=30)
    plt.show()
    anim.save("learning.gif", writer=writer)


def estimated_params():
    fig, ax = plt.subplots()
    ffamily = 'Cambria'
    fsize = 14
    predict_model = PredictModel('temp/inf_eu_24_new.csv', 10)
    params = np.array([2.07060536e-05, 6.89766919e-02, 2.50552352e+04])

    ax.set_title('обучение модели на реальных данных', family=ffamily, fontsize=fsize)
    ax.set_ylabel('новых случаев заражения', family=ffamily, fontsize=fsize)

    plt.xticks(rotation=70)
    ax.plot(predict_model.data.index.values, predict_model.data, label='реальные данные', color='red')

    model = predict_model.sir
    model.set_factors(beta=params[0], gama=params[1])
    model.set_start_stages(S=params[2])

    model.start(len(predict_model.data))
    ax.plot(model.flows_df['F(S>I)'], color='grey', label='модель')
    fig.subplots_adjust(bottom=0.15)
    plt.legend()
    plt.savefig('images_intervals/learning_final.png', dpi=300)
    plt.show()

    print('r2', r2_score(predict_model.data, model.flows_df['F(S>I)'][:-1]))


def real_prognoz():
    fig, ax = plt.subplots()

    train_time = 26
    predict_model = PredictModel('temp/inf_eu_24_new.csv', 10, train_time=train_time)
    # result = minimize(predict_model.get_error, np.array([0.000004, 0.1, 100000]),
                      # bounds=[(0, 1), (0, 1), (10_000, 10_000_000)], method='Nelder-Mead')
    # print(result.x)
    # params = result.x
    params = np.array([2.12429998e-05, 1.10294192e-01, 2.60871081e+04])
    # params = np.array([2.28367255e-05, 7.56167144e-01, 4.97576481e+04])
    # params = np.array([2.07060536e-05, 6.89766919e-02, 2.50552352e+04])
    model = predict_model.sir
    model.set_factors(beta=params[0], gama=params[1])
    model.set_start_stages(S=params[2])
    # model.start(40)

    real = model.sir_with_data_delta(predict_model.data.values.ravel())

    # real.plot()
    ax.plot(predict_model.data.index, real['I'][:-1], color='red', label='реальные данные')

    ss, ii, rr = real.iloc[train_time, :]
    model.set_start_stages(S=ss, I=ii, R=rr)
    progn_time = len(predict_model.data) - train_time
    stoch_st, stoch_fl = model.several_stoch_runs(100, progn_time)

    for p_value in (0.9, 0.95, 0.99):
        intervals = model.confidence_interval(stoch_st, p_value=p_value)
        intervals['I'] = intervals['I'][:-1]
        intervals['I'].index = np.arange(train_time, train_time + progn_time)
        draw_intervals(intervals['I'], p_value=p_value, color='red', ax=ax, label=f'p={p_value}')

    ax.set_ylabel('количество инфицированных', size=14, family='Cambria')
    ax.set_title('доверительный интервал прогнозирования', size=14, family='Cambria')
    fig.subplots_adjust(bottom=0.15)
    # plt.plot(predict_model.train_data, label='flow')
    plt.axvline(train_time, linestyle='dashed', color='black', alpha=0.5)
    plt.xticks(rotation=70)
    plt.legend()
    plt.savefig('images_intervals/prognoz_intervals.png', dpi=300)
    plt.show()


def real_prognoz_fl():
    fig, ax = plt.subplots()

    train_time = 26
    predict_model = PredictModel('temp/inf_eu_24_new.csv', 10, train_time=train_time)
    # result = minimize(predict_model.get_error, np.array([0.000004, 0.1, 100000]),
                      # bounds=[(0, 1), (0, 1), (10_000, 10_000_000)], method='Nelder-Mead')
    # print(result.x)
    # params = result.x
    params = np.array([2.12429998e-05, 1.10294192e-01, 2.60871081e+04])
    # params = np.array([2.28367255e-05, 7.56167144e-01, 4.97576481e+04])
    # params = np.array([2.07060536e-05, 6.89766919e-02, 2.50552352e+04])
    model = predict_model.sir
    model.set_factors(beta=params[0], gama=params[1])
    model.set_start_stages(S=params[2])
    # model.start(40)

    real = model.sir_with_data_delta(predict_model.data.values.ravel())

    # real.plot()
    ax.plot(predict_model.data.index, predict_model.data.values.ravel(), color='red', label='реальные данные')

    ss, ii, rr = real.iloc[train_time, :]
    model.set_start_stages(S=ss, I=ii, R=rr)
    progn_time = len(predict_model.data) - train_time
    stoch_st, stoch_fl = model.several_stoch_runs(100, progn_time)

    for p_value in (0.9, 0.95, 0.99):
        intervals = model.confidence_interval(stoch_fl, p_value=p_value)
        intervals['F(S>I)'] = intervals['F(S>I)'][:-1]
        intervals['F(S>I)'].index = np.arange(train_time, train_time + progn_time)
        draw_intervals(intervals['F(S>I)'], p_value=p_value, color='red', ax=ax, label=f'p={p_value}')

    ax.set_ylabel('новых случаев заражения', size=14, family='Cambria')
    ax.set_title('доверительный интервал прогнозирования', size=14, family='Cambria')
    fig.subplots_adjust(bottom=0.15)
    # plt.plot(predict_model.train_data, label='flow')
    plt.axvline(train_time, linestyle='dashed', color='black', alpha=0.5)
    plt.xticks(rotation=70)
    plt.legend()
    plt.savefig('images_intervals/prognoz_intervals_flows.png', dpi=300)
    plt.show()

def real_intervals():
    ffize = 14
    ffamily = 'Cambria'
    predict_model = PredictModel('temp/inf_eu_24_new.csv', 10)
    # predict_model.predict_show(np.array([2.071e-05,  6.898e-02,  2.506e+04]), 50)
    result = np.array([2.07060536e-05, 6.89766919e-02, 2.50552352e+04])
    model = predict_model.sir
    model.set_factors(beta=result[0], gama=result[1])
    model.set_start_stages(S=result[2])

    time = len(predict_model.data)
    fig, ax = plt.subplots()

    teor = model.start(time)
    real = model.sir_with_data_delta(predict_model.data.values.ravel())
    ax.plot(predict_model.data.index, real['I'][:-1], color='red', label='реальные данные')
    ax.plot(teor['I'], color='red', linestyle='dashed', label='модель')

    stages_ = 'SIR'
    stages = ('susceptible', 'infected', 'recovered')
    stages_ru = ('восприимчивые (S)', 'инфицированные (I)', 'выздоровевшие (R)')
    colors = ('green', 'red', 'blue')
    np.random.seed(2)
    stoch_st, stoch_fl = model.several_stoch_runs(300, time)

    for p_value in (0.9, 0.95, 0.99):
        intervals = model.confidence_interval(stoch_st, p_value=p_value)
        draw_intervals(intervals['I'], p_value=p_value, color='red', ax=ax, label=f'p={p_value}')

    ax.set_xlabel('время (у.е.)', size=ffize, family=ffamily)
    ax.set_ylabel('количество инфицированных',  size=ffize, family=ffamily)
    ax.set_title('доверительный интервал для модели распространения гриппа', size=ffize, family=ffamily)
    plt.xticks(rotation=70)
    ax.grid()
    fig.subplots_adjust(bottom=0.15)
    plt.legend()
    plt.savefig('images_intervals/intervals_real_full.png', dpi=300)
    plt.show()

    # for pvalue, alpha in zip([90, 95, 99], [0.1, 0.2, 0.3]):
    #     st_max = np.percentile(st_stoch, pvalue, axis=1)
    #     st_min = np.percentile(st_stoch, 100 - pvalue, axis=1)
    #     # print(len(st_min), len(st_max))
    #     plt.fill_between(np.arange(time+1), st_min, st_max, alpha=alpha, color='green')
    #
    # teor = model.start(time)
    # real = model.sir_with_data_delta(predict_model.data.values.ravel())
    # plt.plot(real['I'], color='red')
    # plt.plot(teor['I'], linestyle='dashed')
    #
    # plt.show()


def draw_intervals(edges: pd.DataFrame, color, ax: plt.axis, p_value, label="area"):
    alpha = {0.9: 0.3, 0.95: 0.2, 0.99: 0.1}[p_value]
    ax.fill_between(edges.index, edges.iloc[:, 0], edges.iloc[:, 1], color=color, alpha=alpha, label=label)


def sir_with_stoch():
    fsize = 14
    ffamily = "Cambria"

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    beta = 0.0004
    gama = 0.03
    time = 150
    sir1 = EpidemicModel.get_sir(beta=beta, gama=gama)
    res = sir1.start(time)
    np.random.seed(1)

    stages_ = 'SIR'
    stages = ('susceptible', 'infected', 'recovered')
    stages_ru = ('восприимчивые (S)', 'инфицированные (I)', 'выздоровевшие (R)')
    colors = ('green', 'red', 'blue')

    st_stoch, fl_stoch = sir1.several_stoch_runs(time=time, n=60)

    for i, st in enumerate(stages_):
        ax[i].plot(res[st], color=colors[i], label=stages[i])
        ax[i].plot(st_stoch[st].values, color=colors[i], alpha=0.05)
        ax[i].set_title(stages_ru[i], family=ffamily, fontsize=fsize)

    ylim = max([ax[i].get_ylim()[1] for i in range(3)])
    for i in range(3):
        ax[i].set_ylim([0, ylim])
        ax[i].set_xlabel('время (у.е.)', family=ffamily, fontsize=fsize)
        ax[i].grid(which='both', alpha=0.4)

    ax[0].set_xlim([0, 60])

    ax[1].set_yticklabels([])
    ax[1].yaxis.grid(True)
    ax[2].set_yticklabels([])
    ax[2].yaxis.grid(True)

    ax[0].set_ylabel("количество индивидов", family=ffamily, fontsize=fsize,
                     rotation="vertical", labelpad=12)

    fig.subplots_adjust(wspace=0, hspace=0, left=0.1, right=0.99)

    fig.suptitle(f'Моделирование при β = {0.4}, γ = {gama}', family=ffamily, fontsize=fsize)
    plt.savefig('images_intervals/abstract_sir_stoch2.png', dpi=300)
    plt.show()


def sir_with_intervals():
    fsize = 14
    ffamily = "Cambria"

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    beta = 0.0004
    gama = 0.03
    time = 150
    sir1 = EpidemicModel.get_sir(beta=beta, gama=gama)
    res = sir1.start(time)
    np.random.seed(2)

    stages_ = 'SIR'
    stages = ('susceptible', 'infected', 'recovered')
    stages_ru = ('восприимчивые (S)', 'инфицированные (I)', 'выздоровевшие (R)')
    colors = ('green', 'red', 'blue')

    st_stoch, fl_stoch = sir1.several_stoch_runs(time=time, n=100)
    intervals_p = {}
    for p_value in (0.9, 0.95, 0.99):
        intervals_p[p_value] = sir1.confidence_interval(st_stoch, p_value=p_value)

    for i, st in enumerate(stages_):
        ax[i].plot(res[st], color=colors[i], label=stages[i])
        # ax[i].plot(st_stoch[st].values, color=colors[i], alpha=0.05)
        ax[i].set_title(stages_ru[i], family=ffamily, fontsize=fsize)

        for p_value in intervals_p:
            draw_intervals(intervals_p[p_value][st], colors[i], ax[i], p_value, label=f'p={p_value}')

    ylim = max([ax[i].get_ylim()[1] for i in range(3)])
    for i in range(3):
        ax[i].set_ylim([0, ylim])
        ax[i].set_xlabel('время (у.е.)', family=ffamily, fontsize=fsize)
        ax[i].grid(which='both', alpha=0.4)
        ax[i].legend()

    ax[0].set_xlim([0, 60])

    ax[1].set_yticklabels([])
    ax[1].yaxis.grid(True)
    ax[2].set_yticklabels([])
    ax[2].yaxis.grid(True)

    ax[0].set_ylabel("количество индивидов", family=ffamily, fontsize=fsize,
                    rotation="vertical", labelpad=12)

    fig.subplots_adjust(wspace=0, hspace=0, left=0.1, right=0.99)
    # ylim = max([ax[i][j].get_ylim()[1] for j in range(num_col) for i in range(num_row)])
    # for j in range(num_col):
    #     for i in range(num_row):
    #         ax[i][j].set_ylim([0, ylim])

    # st_intervals = sir1.confidence_interval(st_stoch, p_value=0.9)
    # draw_intervals(st_intervals['I'], 'red', ax, 0.9, label='I (p=0.9)')
    #
    # st_intervals = sir1.confidence_interval(st_stoch, p_value=0.99)
    # draw_intervals(st_intervals['S'], 'green', ax, 0.99)


    # draw_intervals(st_intervals['I'], 'red', ax, 0.99)
    # draw_intervals(st_intervals['R'], 'blue', ax, 0.99)

    # sir1.print_result_table()
    # plt.legend()
    fig.suptitle(f'Моделирование при β = {0.4}, γ = {gama}', family=ffamily, fontsize=fsize)
    plt.savefig('images_intervals/abstract_sir_intervals2.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    # sir_exp2()
    # estimate_params()
    estimated_params()
    # sir_with_stoch()
    # sir_with_intervals()
    # real_intervals()
    # real_prognoz_fl()