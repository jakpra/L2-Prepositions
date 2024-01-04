import pandas as pd
import torch
from scipy import stats
import math


def approx_rand(model_guide, data, column, r=10000, mappings=None):
    # est = model_guide(data)['outcomes']
    est = {}

    values = set(data[column].values)
    seen = set()
    for v1 in values:
        # v1_mask = list(data[column] == v1)
        # data1 = data[v1_mask]
        seen.add(v1)
        for v2 in values:
            if v2 not in seen:
                # v2_mask = list(data[column] == v2)
                # data2 = data[v2_mask]

                if mappings:
                    _v1 = mappings[column]['i2v'][str(v1)]
                    _v2 = mappings[column]['i2v'][str(v2)]
                print(_v1, _v2, end=' ')

                # if len(est1) == len(est2):
                #     pass
                # else:
                #     # print(mappings[column]['i2v'][str(v1)], len(est1), mappings[column]['i2v'][str(v2)], len(est2))
                #     continue
                # swapped_data1 = data1.copy()
                # swapped_data1.loc[:, column] = v2
                # swapped_data2 = data2.copy()
                # swapped_data2.loc[:, column] = v1
                #
                # assert not (swapped_data1 == data1).all().all()

                # est1 = model_guide(pd.concat((data1, swapped_data2)))['outcomes']
                # est2 = model_guide(pd.concat((swapped_data1, data2)))['outcomes']
                if v1 not in est:
                    v1_data = data.copy()
                    v1_data.loc[:, column] = v1
                    est[v1] = model_guide(v1_data)['outcomes']

                est1 = est[v1]

                if v2 not in est:
                    v2_data = data.copy()
                    v2_data.loc[:, column] = v2
                    est[v2] = model_guide(v2_data)['outcomes']

                est2 = est[v2]

                # swapped_est1 = model_guide(swapped_data1)['condition_effect']
                # swapped_est2 = model_guide(swapped_data2)['condition_effect']
                # print(est1.shape)
                # print(data1['outcomes'][:10])
                # assert not (est1 == data1['outcomes']).all()
                # assert (swapped_est1 == est1).all()

                real_diff = (est1.mean() - est2.mean()).abs()
                # print('real', est1.mean(), est2.mean(), real_diff)
                c = 0
                for _ in range(r):

                    shuffle1 = torch.randint(2, est1.shape).bool()
                    # shuffle2 = torch.randint(2, est2.shape).bool()

                    # print(shuffle1)
                    # print(shuffle2)

                    _est1 = est1.where(shuffle1, est2)
                    _est2 = est2.where(shuffle1, est1)

                    pseudo_diff = (_est1.mean() - _est2.mean()).abs()
                    if pseudo_diff >= real_diff:
                        # print('pseudo', _est1.mean(), _est2.mean(), pseudo_diff)
                        c += 1

                p = (c + 1) / (r + 1)
                # if mappings:
                #     _v1 = mappings[column]['i2v'][str(v1)]
                #     _v2 = mappings[column]['i2v'][str(v2)]
                # print(_v1, _v2, p)
                print(p)
                print()


def pooled_var(sd1, n1, sd2, n2):
    return ((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2)


def cohens_d(mu1, sd1, n1, mu2, sd2, n2):
    if n2 == 0:
        return mu1 / sd1
    return (mu1 - mu2) / math.sqrt(pooled_var(sd1, n1, sd2, n2))


def hedges_g(mu1, sd1, n1, mu2, sd2, n2):
    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    return cohens_d(mu1, sd1, n1, mu2, sd2, n2) * correction


def mean_diff_conf_inter(mu1, sd1, n1, mu2, sd2, n2, q=.95, t=None):
    if t is None:
        t = stats.t.ppf(q=(1 - ((1 - q) / 2)), df=(n1 - 1 + ((n2 - 1) if n2 > 0 else 0)))
    var = pooled_var(sd1, n1, sd2, n2) if n2 > 0 else sd1
    a = (mu1 - mu2) if n2 > 0 else mu1
    b = t * (math.sqrt((var / n1) + (var / n2)) if n2 > 0 else sd1)
    return a - b, a + b


def ci_p(ci, t, df):
    lo, hi = ci
    est = abs(lo + hi) / 2
    se = (hi - lo) / (2 * t)
    z = est / se
    # print(est)
    # print(se)
    # print(z)
    return 2 * (1 - stats.t.cdf(z, df))


def mean_diff_p(mu1, sd1, n1, mu2, sd2, n2, q_ci=0.95, q_p=0.95):
    df = n1 + n2 - 2
    t = stats.t.ppf(q=(1 - ((1 - q_p) / 2)), df=df)
    if q_ci != q_p:
        ci = mean_diff_conf_inter(mu1, sd1, n1, mu2, sd2, n2, q=q_ci)
    else:
        ci = mean_diff_conf_inter(mu1, sd1, n1, mu2, sd2, n2, t=t)
    return ci_p(ci, t, df)
