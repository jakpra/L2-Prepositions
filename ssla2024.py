import sys
import json
import glob
import pandas as pd
from collections import defaultdict, Counter

import math
import random
from itertools import combinations, chain

import torch
import pyro.infer
import pyro.optim

from scipy import stats

import brmp
from cprior.cdist import ci_interval, ci_interval_exact

from significance import mean_diff_conf_inter, mean_diff_p, cohens_d, pooled_var
from util import *


###########################################################################


runs = 1
global_seed = 333
n_svi_iter = 1000
n_samples = 100
split_fracs = [0.84, 0.01, 0.15]  # train, dev, test
no_ctrl = True
no_dly = True


###########################################################################



def summarize(fit):
    m = fit.marginals()  # qs=[0.001, 0.005, 0.01, 0.9, 0.95, 0.99])
    means = dict(zip(m.row_labels, torch.FloatTensor(m.array[:, 0])))
    sds = dict(zip(m.row_labels, torch.FloatTensor(m.array[:, 1])))
    n_effs = dict(zip(m.row_labels, torch.FloatTensor(m.array[:, -2])))
    result_means = {}
    result_sds = {}
    result_n_effs = {}
    interesting = {}

    mappings = defaultdict(lambda: defaultdict(dict))
    for k, mean in sorted(means.items()):
        if k.startswith('r_'):
            f_id, i = k[2:].strip(']').split('[')
            i, fixed = i.split(',')
            fixed = fixed.replace(':', 'x')
            fixed_f_id = f'{fixed}:{f_id}'
            fs = f_id.split(':')
            _is = i.split('_')
            for _f, _i in zip(fs, _is):
                if _f not in mappings:
                    mappings[_f] = {'v2i': {}, 'i2v': {}}
                if _i not in mappings[_f]['v2i']:
                    mapped = len(mappings[_f]['v2i'])
                    mappings[_f]['v2i'][_i] = mapped
                    mappings[_f]['i2v'][str(mapped)] = _i
            if fixed not in mappings:
                mappings[fixed] = {'v2i': {fixed: 0}, 'i2v': {'0': fixed}}

    for k, mean in sorted(means.items()):
        sd = sds[k]
        n_eff = n_effs[k]
        if k.startswith('sd_'):
            f_id, fixed = k[3:].split('__')
            fixed = fixed.replace(':', 'x')
            fixed_f_id = f'{fixed}:{f_id}'
            if len(f_id.split(':')) == 1 or mean > 0.02:
                interesting[fixed_f_id] = mean
                if mean > interesting.get(fixed_f_id, 0):
                    interesting[fixed_f_id] = mean
        elif k.startswith('b_'):
            f_id = k[2:]
            if abs(mean) > 0.02:
                interesting[f_id] = mean
            result_means[f_id] = mean
            result_sds[f_id] = sd
            result_n_effs[f_id] = n_eff
        elif k.startswith('r_'):
            f_id, i = k[2:].strip(']').split('[')
            i, fixed = i.split(',')
            fixed = fixed.replace(':', 'x')
            fixed_f_id = f'{fixed}:{f_id}'
            fs = f_id.split(':')
            _is = i.split('_')
            try:
                _is = list(map(int, _is))
            except ValueError:
                _is = list(map(lambda x: mappings[x[0]]['v2i'].get(x[1]), zip(fs, _is)))
            _is = tuple(map(torch.tensor, _is))
            if fixed_f_id not in result_means:
                ns = [len(mappings[_f]['v2i']) for _f in fs]
                result_means[fixed_f_id] = torch.zeros(*ns)
                result_sds[fixed_f_id] = torch.zeros(*ns)
                result_n_effs[fixed_f_id] = torch.zeros(*ns)

            result_means[fixed_f_id].index_put_(_is, mean)
            result_sds[fixed_f_id].index_put_(_is, sd)
            result_n_effs[fixed_f_id].index_put_(_is, n_eff)

    return result_means, result_sds, result_n_effs, interesting, mappings




def get_condition_from_path(path):
    flag = False
    for _cond in ['F2F', 'HalfHalf', 'CM', 'TRAD', 'CORR', 'Control', 'CL']:
        if _cond.lower() in path.lower():
            cond = _cond
            flag = True
            break
    if not flag: return None
    flag = False
    for _time in ['Pre', 'Post', 'Delay']:
        if _time.lower() in path.lower():
            time = _time.upper()
            flag = True
            break
    if not flag: return None
    flag = False
    for _task in ['Cloze', 'Translation', 'PET', 'GJT']:
        if _task.lower() in path.lower():
            task = _task.lower()
            flag = True
            break
    if not flag: return None

    if 'LSKC' in path.upper():
        school = 'LSKC'
    elif 'SFXC' in path.upper():
        school = 'SFXC'
    else:
        school = 'NONE'

    return {'condition_id': cond, 'time_id': time, 'task_id': task, 'answer_id': task, 'school_id': school}


data_dir = sys.argv[1]

correct_count = Counter()
total_count = Counter()

summary = pd.DataFrame(columns=['task_id', 'answer_id', 'time_id', 'condition_id', 'outcomes'])

for filename in glob.glob(data_dir):
    # if not filename.endswith('_final.csv'): continue
    condition = get_condition_from_path(filename)
    if not condition: continue
    if no_ctrl and condition['condition_id'] == 'Control': continue
    if no_dly and condition['time_id'] == 'DELAY': continue
    print(condition)

    df = pd.read_csv(filename)

    if 'Rule Name' not in df.columns:
        if 'Sense Name' in df.columns:
            df['Rule Name'] = df['Sense Name']

    df = df[~df['Rule Name'].isnull()]
    df = df[df['Rule Name'] != 'Distractor']
    df = df[~df['Rule Name'].str.contains('Filler')]

    df['target'] = df.apply(lambda x: x['Rule Name'].split('-')[0].lower(), axis=1)

    if condition['task_id'] == 'translation':
        df = df[df['Response Type'] == 'Enter Key']

    n = len(df)

    new_data = {k: [v] * n for k, v in condition.items()}

    new_data['outcomes'] = df['Correct']
    new_data['fxn_id'] = df.apply(lambda x: x['Rule Name'].rsplit('-', maxsplit=1)[0].lower(), axis=1)
    new_data['idiom_id'] = df.apply(lambda x: x['Rule Name'].split('-')[-1].lower(), axis=1)
    new_data['student_id'] = df['Participant Private ID']
    new_data['item_id'] = df['Sentence No.']

    summary = pd.concat([summary, pd.DataFrame(new_data)])

print()


all_data = data = summary.astype('category')

n = len(data)

settings = [
    # ('all',
    #  ['condition_id', 'time_id', 'answer_id', 'fxn_id', 'idiom_id', 'student_id', 'school_id']),
    # ('cloze',
    #  ['condition_id', 'time_id', 'fxn_id', 'idiom_id', 'student_id', 'school_id']),
    # ('trans',
    #  ['condition_id', 'time_id', 'fxn_id', 'idiom_id', 'student_id', 'school_id']),
    # ('all_noproficiency',
    #  ['condition_id', 'time_id', 'answer_id', 'fxn_id', 'idiom_id']),
    ('cloze_noproficiency',
     ['condition_id', 'time_id', 'fxn_id', 'idiom_id']),
    # ('trans_noproficiency',
    #  ['condition_id', 'time_id', 'fxn_id', 'idiom_id']),
    ('pet_noproficiency',
     ['condition_id', 'time_id', 'fxn_id', 'idiom_id']),
    # ('all_noproficiency_noprep',
    #  ['condition_id', 'time_id', 'answer_id', 'idiom_id']),
    # ('cloze_noproficiency_noprep',
    #  ['condition_id', 'time_id', 'idiom_id']),
    # # ('trans_noproficiency',
    # #  ['condition_id', 'time_id', 'fxn_id', 'idiom_id']),
    # ('pet_noproficiency_noprep',
    #  ['condition_id', 'time_id', 'idiom_id']),
]

plot_data = {}

for setting_name, factors_to_include in settings:
    contrasts = None
    fixed_eff = '1'
    if 'lm_tgt_p' in factors_to_include:
        fixed_eff += ' + lm_tgt_p'
    if 'lm_avg_p' in factors_to_include:
        fixed_eff += ' + lm_avg_p'
        if 'lm_tgt_p' in factors_to_include:
            fixed_eff += ' + lm_tgt_p:lm_avg_p'

    interaction_factors = [fac for fac in factors_to_include if
                           fac not in {'student_id', 'lm_tgt_p', 'lm_avg_p', 'lm_hidden'}]

    formula = f'outcomes ~ {fixed_eff}'
    if setting_name == 'gjt-irt':
        formula += ' + (1 || student_id:fxn_id:idiom_id:answer_id) + (1 || condition_id:time_id) + (1 || lm_hidden)'
    else:
        if 'student_id' in factors_to_include:
            formula += f' + ({fixed_eff} || student_id)'
            for factor in interaction_factors:
                if factor not in ('time_id', 'condition_id', 'school_id'):
                    formula += f' + ({fixed_eff} || student_id:{factor})'

        if 'lm_hidden' in factors_to_include:
            formula += ' + (1 || lm_hidden)'

        if interaction_factors:
            interaction_ns = list(range(2, len(interaction_factors) + 1))
            random_effs = [":".join(tup) for tup in get_tuples(interaction_factors, interaction_ns)]
            formula += (' + ' + ' + '.join(f'(1 || {rand_e})' for rand_e in interaction_factors)
                        # + ' + ' + ' + '.join(f'{rand_e}' for rand_e in random_effs))
                        + ' + ' + ' + '.join(f'({fixed_eff} || {rand_e})' for rand_e in random_effs))

    print()
    print('\n SETTING NAME:')
    print(setting_name)
    print('\n FACTORS:')
    print(factors_to_include)
    print('\n FORMULA:')
    print(formula)

    cloze_data = all_data[all_data['task_id'] == 'cloze'].copy()
    print(f'task=Cloze data len', len(cloze_data))
    trans_data = all_data[all_data['task_id'] == 'translation'].copy()
    print(f'task=Translation data len', len(trans_data))
    pet_data = all_data[all_data['task_id'] == 'pet'].copy()
    print(f'task=PET data len', len(pet_data))

    if 'cloze' in setting_name:
        data = cloze_data
    elif 'trans' in setting_name:
        data = trans_data
    elif 'pet' in setting_name:
        data = pet_data
    else:
        data = all_data

    for N in range(runs):

        SEED = global_seed if global_seed is not None else N  # random.randint(1, 9999)
        print(f'\n\nN={N}, SEED={SEED}')

        random.seed(SEED)
        pyro.set_rng_seed(SEED)
        pyro.clear_param_store()

        data = data.sample(frac=1, random_state=SEED)

        optim = pyro.optim.AdamW({'lr': 5e-2})  # , 'betas': [0.8, 0.99]})
        autoguide = pyro.infer.autoguide.AutoNormal

        model_and_data = brmp.brm(formula, data, family=brmp.family.Bernoulli,
                                  priors=[
                                      brmp.priors.Prior(('b',), brmp.family.Normal(0., 1.)),
                                      brmp.priors.Prior(('sd',), brmp.family.HalfNormal(3.)),

                                      # brmp.priors.Prior(('b', 'condition_id[CL]'), brmp.family.Normal(0.7408, 0.1244)),
                                      # brmp.priors.Prior(('b', 'condition_id[TRAD]'), brmp.family.Normal(0.7106, 0.1494)),
                                      # brmp.priors.Prior(('b', 'condition_id[CORR]'), brmp.family.Normal(0.6795, 0.1239)),
                                      # brmp.priors.Prior(('b', 'condition_id[Control]'), brmp.family.Normal(0.5660, 0.1406)),
                                  ],
                                  contrasts=None if contrasts is None else {k: v.numpy() for k, v in contrasts.items()}
                                  )

        pyro.clear_param_store()

        with open(f'brmp_pyro_model_{setting_name}.py', 'w') as f:
            f.write(brmp.pyro_codegen.genmodel(model_and_data.model.desc))

        fit = model_and_data.svi(iter=n_svi_iter, num_samples=n_samples, seed=SEED, optim=optim, autoguide=autoguide)
        print('\nALL MARGINALS')
        print(fit.marginals())
        print()

        means, sds, n_effs, interesting, mappings = summarize(fit=fit)
        setting_signi = {}
        setting_plot_data = {}
        print('SIGNIFICANCE')
        for name_i, name in enumerate(interesting):
            print()
            setting_signi[name] = defaultdict(dict)
            print(setting_name)
            if means[name].numel() == 1:
                diff_name = f'{name} > 0'
                mu, sd, n_eff = means[name], sds[name], n_effs[name]
                conf_inter = mean_diff_conf_inter(mu, sd, n_samples, 0, 0, 0, q=0.95)
                dist = stats.norm(mu, sd)
                cred_inter_eti_95 = ci_interval_exact(dist, interval_length=0.95, method='ETI')
                cred_inter_hdi_95 = ci_interval_exact(dist, interval_length=0.95, method='HDI')
                cred_inter_eti_89 = ci_interval_exact(dist, interval_length=0.89, method='ETI')
                cred_inter_hdi_89 = ci_interval_exact(dist, interval_length=0.89, method='HDI')
                eff_size = cohens_d(mu, sd, n_samples, 0, 0, 0)
                p_val = mean_diff_p(mu, sd, n_samples, 0, 0, 0)
                setting_signi[name][0][0] = {'p': p_val, 'd': eff_size, 'ci': conf_inter,
                                             'cred_inter_eti_95': cred_inter_eti_95,
                                             'cred_inter_hdi_95': cred_inter_hdi_95,
                                             'cred_inter_eti_89': cred_inter_eti_89,
                                             'cred_inter_hdi_89': cred_inter_hdi_89
                                             }
                print(name, f'mu={mu:.2f}', f'sd={sd:.2f}', f'n_eff={n_eff:.2f}')
                print(diff_name,
                      f'CredIdiff_hdi_95={cred_inter_hdi_95[0]:.3f};{cred_inter_hdi_95[1]:.3f}',
                      f'CredIdiff_hdi_89={cred_inter_hdi_89[0]:.3f};{cred_inter_hdi_89[1]:.3f}',
                      f'Cohen\'s d={eff_size:.4f}',
                      'p is NaN' if math.isnan(p_val) else f'p={p_val:.4f}' if p_val >= 0.0001 else 'p<0.0001',
                      '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else '--')
                setting_plot_data[name] = {'mu': [mu.item()], 'sd': [sd.item()], 'name': [name]}
                continue
            print(f'{name} max sd', f'{interesting[name]:.2f}')
            idxs = sds[name].nonzero().tolist()
            fs = name.split(':')
            itr = sorted(zip(means[name].view(-1), idxs, sds[name].view(-1), n_effs[name].view(-1)), reverse=True)
            plot_mus, plot_names, plot_sds = [], [], []
            for i, vals1 in enumerate(itr):
                mu, idx, sd, n_eff = vals1
                idx_name = ':'.join([f"{mappings[f]['i2v'][str(_idx)]}" for _idx, f in zip(idx, fs[1:])])  # f"{mappings[f]['i2v'][str(_idx)]}"
                plot_mus.append(mu.item())
                plot_names.append(idx_name)
                plot_sds.append(sd.item())
                if 'student_id' in name or 'lm_hidden' in name:
                    continue
                print(f'{name}={idx_name}', f'mu={mu:.2f}', f'sd={sd:.2f}', f'n_eff={n_eff:.2f}')
                for vals2 in itr[i + 1:]:
                    (mu1, idx1, sd1, n1), (mu2, idx2, sd2, n2) = sorted([vals1, vals2], reverse=True)
                    idx_diff = sum([int(i2 != i1) for i1, i2 in zip(idx1, idx2)])
                    if idx2 != idx1 and mu1 != mu2 and idx_diff == 1:
                        idx1_name = ';'.join([f"{f}={mappings[f]['i2v'][str(i1)]}" for i1, f in zip(idx1, fs[1:])])  # mappings[f]['i2v'][str(i1)]
                        idx2_name = ';'.join([f"{f}={mappings[f]['i2v'][str(i2)]}" for i2, f in zip(idx2, fs[1:])])  # mappings[f]['i2v'][str(i2)]
                        diff_name = f'{idx1_name} > {idx2_name}'
                        conf_inter = mean_diff_conf_inter(mu1, sd1, n_samples, mu2, sd2, n_samples, q=0.95)
                        dist1 = stats.norm(mu1, sd1)
                        dist2 = stats.norm(mu2, sd2)
                        cred_inter1_eti_95 = ci_interval_exact(dist1, interval_length=0.95, method='ETI')
                        cred_inter2_eti_95 = ci_interval_exact(dist2, interval_length=0.95, method='ETI')
                        cred_inter1_hdi_95 = ci_interval_exact(dist1, interval_length=0.95, method='HDI')
                        cred_inter2_hdi_95 = ci_interval_exact(dist2, interval_length=0.95, method='HDI')
                        cred_inter1_eti_89 = ci_interval_exact(dist1, interval_length=0.89, method='ETI')
                        cred_inter2_eti_89 = ci_interval_exact(dist2, interval_length=0.89, method='ETI')
                        cred_inter1_hdi_89 = ci_interval_exact(dist1, interval_length=0.89, method='HDI')
                        cred_inter2_hdi_89 = ci_interval_exact(dist2, interval_length=0.89, method='HDI')
                        diff_dist = stats.norm(mu1-mu2, math.sqrt(pooled_var(sd1, n_samples, sd2, n_samples)))
                        cred_inter_eti_95 = ci_interval_exact(diff_dist, interval_length=0.95, method='ETI')
                        cred_inter_hdi_95 = ci_interval_exact(diff_dist, interval_length=0.95, method='HDI')
                        cred_inter_eti_89 = ci_interval_exact(diff_dist, interval_length=0.89, method='ETI')
                        cred_inter_hdi_89 = ci_interval_exact(diff_dist, interval_length=0.89, method='HDI')
                        eff_size = cohens_d(mu1, sd1, n_samples, mu2, sd2, n_samples)
                        p_val = mean_diff_p(mu1, sd1, n_samples, mu2, sd2, n_samples)
                        setting_signi[name][idx1_name][idx2_name] = {'p': p_val, 'd': eff_size, 'ci': conf_inter,
                                                                     'cred_inter1_eti_95': cred_inter1_eti_95,
                                                                     'cred_inter2_eti_95': cred_inter2_eti_95,
                                                                     'cred_inter1_hdi_95': cred_inter1_hdi_95,
                                                                     'cred_inter2_hdi_95': cred_inter2_hdi_95,
                                                                     'cred_inter_diff_eti_95': cred_inter_eti_95,
                                                                     'cred_inter_diff_hdi_95': cred_inter_hdi_95,
                                                                     'cred_inter1_eti_89': cred_inter1_eti_89,
                                                                     'cred_inter2_eti_89': cred_inter2_eti_89,
                                                                     'cred_inter1_hdi_89': cred_inter1_hdi_89,
                                                                     'cred_inter2_hdi_89': cred_inter2_hdi_89,
                                                                     'cred_inter_diff_eti_89': cred_inter_eti_89,
                                                                     'cred_inter_diff_hdi_89': cred_inter_hdi_89
                                                                     }
                        setting_signi[name][idx2_name][idx1_name] = {'p': p_val, 'd': eff_size, 'ci': conf_inter,
                                                                     'cred_inter1_eti_95': cred_inter1_eti_95,
                                                                     'cred_inter2_eti_95': cred_inter2_eti_95,
                                                                     'cred_inter1_hdi_95': cred_inter1_hdi_95,
                                                                     'cred_inter2_hdi_95': cred_inter2_hdi_95,
                                                                     'cred_inter_diff_eti_95': cred_inter_eti_95,
                                                                     'cred_inter_diff_hdi_95': cred_inter_hdi_95,
                                                                     'cred_inter1_eti_89': cred_inter1_eti_89,
                                                                     'cred_inter2_eti_89': cred_inter2_eti_89,
                                                                     'cred_inter1_hdi_89': cred_inter1_hdi_89,
                                                                     'cred_inter2_hdi_89': cred_inter2_hdi_89,
                                                                     'cred_inter_diff_eti_89': cred_inter_eti_89,
                                                                     'cred_inter_diff_hdi_89': cred_inter_hdi_89}
                        print(diff_name,
                              f'{mu1:.3f} +- {sd1:.3f} > {mu2:.3f} +- {sd2:.3f}',
                              f'diff={mu1 - mu2:.3f}',
                              f'CredIs_hdi_95=[{", ".join([f"{x:.3f}" for x in cred_inter1_hdi_95])}] > '
                              f'[{", ".join([f"{x:.3f}" for x in cred_inter2_hdi_95])}]',
                              f'CredIs_hdi_89=[{", ".join([f"{x:.3f}" for x in cred_inter1_hdi_89])}] > '
                              f'[{", ".join([f"{x:.3f}" for x in cred_inter2_hdi_89])}]',
                              f'CredIdiff_hdi_95={cred_inter_hdi_95[0]:.3f};{cred_inter_hdi_95[1]:.3f}',
                              f'CredIdiff_hdi_89={cred_inter_hdi_89[0]:.3f};{cred_inter_hdi_89[1]:.3f}',
                              f'ConfI={conf_inter[0]:.3f};{conf_inter[1]:.3f}',
                              f'Cohen\'s d={eff_size:.3f}',
                              'p is NaN' if math.isnan(p_val) else f'p={p_val:.4f}' if p_val >= 0.0001 else 'p<0.0001',
                              '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else '--')
                print()
            setting_plot_data[name] = {'mu': plot_mus, 'sd': plot_sds, 'name': plot_names}
            print()
        plot_data[setting_name.split('_')[0]] = setting_plot_data

try:
    with open(f'plot_data_{"_".join(plot_data.keys())}_{global_seed}{"_noctrl" if no_ctrl else ""}.json', 'w') as f:
        json.dump(plot_data, f)
except Exception as e:
    print('failed to write plot data', e, file=sys.stderr)
    pass


