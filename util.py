import math
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from itertools import combinations, chain
import torch

def train_test_split(data, fracs, seed):
    assert sum(fracs) == 1
    result = []
    denom = 1
    rest = data
    for fr in fracs:
        _fr = fr / denom
        if _fr >= 1:
            result.append(rest)
            break
        spl = rest.sample(frac=_fr, random_state=seed)
        result.append(spl)
        denom -= fr
        rest = rest.drop(spl.index)
    return result


def df_eq(df1, df2):
    try:
        return (df1 == df2).all().all()
    except ValueError:
        return False


def predict(d, fit, what='expectation', n_samples=None, model_and_train_data=None):
    f = fit.fitted(what=what, data=None if model_and_train_data is not None and df_eq(d, model_and_train_data.df) else d)
    if n_samples is not None:
        f = f[:n_samples]
    return {'outcomes': torch.tensor(f)}


def get_tuples(items, lengths=[2]):
    return list(chain(*[list(combinations(items, i)) for i in lengths]))


def normal_pdf(x, mu=0, sigma=1):
    sqrt_two_pi = math.sqrt(math.pi * 2)
    return math.exp((((x - mu) / sigma) ** 2) / -2) / (sqrt_two_pi * sigma)


def normal_plot(mu=0, sigma=1, size=6):
    x = np.arange(mu-(size * sigma), mu+(size * sigma), ((sigma * size) + 1) / 100)
    y = np.array([normal_pdf(_x, mu, sigma) for _x in x])
    return x, y


def plot_univariate(means, sds=None, ax=None, labels=None):
    for mu, sigma, label in zip(means, torch.zeros_like(means) if sds is None else sds, labels):
        x, y = normal_plot(mu, sigma)
        sns.lineplot(x=x, y=y, ax=ax, label=label)


def plot_errorbar(means, sds=None, ax=None, labels=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1)
    ax.errorbar(x=means, y=labels, xerr=sds, ls='', elinewidth=0.5, **kwargs)


def visualize(means, effect_name, sds=None, ax=None, **kwargs):
    plot_errorbar(means, sds, ax=ax, labels=effect_name, **kwargs)


def pred_from_probs(probs, random_if_random=True):
    if random_if_random:
        probs = probs.where(probs != 0.5, torch.rand_like(probs))
    return (probs >= 0.5).long()
