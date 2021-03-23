import itertools
import os
import random
import pandas as pd
import numpy as np
import torch

method_names = {'word2vec': 'Word2vec C', 'summation_dot_product': 'DIVE \u0394S * C ', 'dot_product': 'DIVE C',
                'rnd': 'random', 'summation': 'DIVE \u0394S', 'summation_word2vec': 'DIVE \u0394S * Word2vec C',
                'all_subword mean_positional_rank': 'BERT Mean Pos Rank', 'all_subword min_positional_rank': 'BERT '
                                                                                                             'Min Pos '
                                                                                                             'Rank',
                'all_subword max_pattern': 'BERT Max Pattern', 'all_subword mean_pattern': 'BERT Mean Pattern',
                'min score_final_log(z)': 'BERT Min Pos Rank (log(z))', 'min score_final_norm': 'BERT Min Pos Rank (/ '
                                                                                                'norm)',
                'mean score_final_log(z)': 'BERT Mean Pos Rank (log(z))', 'mean score_final_norm': 'BERT Mean Pos '
                                                                                                   'Rank (/ norm)',
                'min bert_soma_total': 'BERT Min Sum', 'mean bert_soma_total': 'BERT Mean Sum'}

# Melhores padrões HypeNet
best_pattern_HypeNet_train_logz = ['{} or some other {}', '{} or any other {}', '{} and any other {}',
                                   '{} is a type of {}', '{} which is kind of {}', '{} and some other {}',
                                   '{} is a {}', '{} a special case of {}', '{} which is a example of {}',
                                   '{} and others {}', '{} which is called {}', '{} or others {}',
                                   '{} which is a class of {}', '{} , a {}', '{} including {}']

best_pattern_HypeNet_train_bert_soma_total = ['{} or some other {}', '{} and any other {}', '{} or any other {}',
                                              '{} is a type of {}', '{} and some other {}', '{} which is kind of {}',
                                              '{} is a {}', '{} or others {}', '{} and others {}', '{} which is a '
                                                                                                   'example of {}',
                                              '{} a special case of {}', '{} , a {}', '{} which is called {}',
                                              '{} which is a class of {}', '{} including {}']


# if combination
# [[hipo 1st sen], [hipo 2nd sen], [hyper 1st sen], [hyper 2nd sen]]
# else
# [[hipo 1st sen], [hyper 1st sen]]

def create_dataframe(json_dict, combination=False, separator=""):
    dict_values = {'hiponimo': [], 'hiperonimo': [], 'classe': [], 'fonte': [], 'pattern': [], 'soma_hipo': [],
                   'soma_hiper': [], 'len_hipo': [], 'len_hiper': []}
    for data, values in json_dict.items():
        hipo, hiper, classe, fonte = data.strip().split(separator)
        for pattern, score in values.items():
            dict_values['hiponimo'].append(hipo)
            dict_values['hiperonimo'].append(hiper)
            dict_values['classe'].append(classe)
            dict_values['fonte'].append(fonte)
            dict_values['pattern'].append(pattern)
            if combination:
                soma = sum(score[0]) + sum(score[1])
            else:
                soma = sum(score[0])
            dict_values['soma_hipo'].append(soma)
            if combination:
                soma = sum(score[2]) + sum(score[3])
            else:
                soma = sum(score[1])
            dict_values['soma_hiper'].append(soma)
            if combination:
                dict_values['len_hipo'].append(len(score[0]))
                dict_values['len_hiper'].append(len(score[2]))
            else:
                dict_values['len_hipo'].append(len(score[0]))
                dict_values['len_hiper'].append(len(score[1]))

    df = pd.DataFrame(dict_values)
    df['bert_soma_total'] = df['soma_hipo'] + df['soma_hiper']
    df['len_total'] = df['len_hipo'] + df['len_hiper']
    return df

def create_dataframe_maskAll(json_dict, separator=""):

    dict_values = {'hiponimo': [], 'hiperonimo': [], 'classe': [], 'fonte': [], 'pattern': [],
                   'len_hipo': [], 'len_hiper': [], 'len_total': [], 'bert_soma_total': []}
    for data, values in json_dict.items():
        hipo, hiper, classe, fonte = data.strip().split(separator)
        len_pair = values['comprimento']
        len_total = sum(len_pair)
        for pattern, score in values.items():
            if pattern == "comprimento":
                continue
            dict_values['hiponimo'].append(hipo)
            dict_values['hiperonimo'].append(hiper)
            dict_values['classe'].append(classe)
            dict_values['fonte'].append(fonte)
            dict_values['len_total'].append(len_total)
            dict_values['pattern'].append(pattern)
            dict_values['len_hipo'].append(len_pair[0])
            dict_values['len_hiper'].append(len_pair[1])
            dict_values['bert_soma_total'].append(sum(score))
    return pd.DataFrame(dict_values)

def filter_by_vocab(path_vocab, dict_data):
    new_data = {}
    vocab = []
    for line in open(path_vocab):
        w, c = line.strip().split()
        vocab.append(w)
    vocab = set(vocab)

    for k, v in dict_data.items():
        hipo, hyper, _, _ = k.strip().split()
        if hipo in vocab and hyper in vocab:
            new_data[k] = v.copy()
    return new_data


# devolve df de balanceamento True/False por tamanho de subtoken
def balanceamento(df, len_size, patterns):
    df_rate = df[df['pattern'] == patterns[0]][['hiponimo', 'hiperonimo', 'classe', 'fonte', 'len_total']]
    df_rate = df_rate.groupby(['len_total'])['fonte'].value_counts()
    error_list = []
    dict_values = {'len_total': [], 'true': [], 'false': []}
    for v in len_size:
        if v in df_rate:
            dict_values['len_total'].append(v)

            if "hyper" in df_rate[v]:
                true_num = df_rate[v]['hyper']
                dict_values['true'].append(true_num)
            else:
                true_num = 0
                dict_values['true'].append(true_num)
            # false é o resto
            false_num = df_rate[v].sum() - true_num
            dict_values['false'].append(false_num)

        else:
            error_list.append(v)

    if error_list:
        print(f"Balanceamento: {error_list} não está no dataframe!")
    df_taxa = pd.DataFrame(dict_values)
    # return df_taxa
    df_taxa['ratio'] = df_taxa['true'] / (df_taxa['true'] + df_taxa['false'])
    return df_taxa


# logsumexp para cada tamanho subtoken e normalização
def logsumexp_normalization(df_data, len_list, pattern_list):
    df = df_data.copy()
    log_store = {}
    logsumexp_store = {}
    normalization = {}
    for size in len_list:
        log_store[size] = {}
        logsumexp_store[size] = {}
        for p in pattern_list:
            if p in log_store:
                raise ValueError
            values = df[(df.pattern == p) & (df.len_total == size)]
            log_store[size][p] = torch.tensor(values['bert_soma_total'].tolist())
            logsumexp_store[size][p] = torch.logsumexp(log_store[size][p], dim=0)

    df['log(Z)'] = df.apply(lambda row: logsumexp_store[row['len_total']][row['pattern']].item(), axis=1)
    # df['sum_bert_by_tokensize'] = df.apply(lambda row: normalization[row['len_total']][row['pattern']].item(), axis=1)
    # score final soma_total - log(Z)
    df['score_final_log(z)'] = df['bert_soma_total'] - df['log(Z)']
    # df['score_final_norm'] = df['bert_soma_total'] / df['sum_bert_by_tokensize']
    return df

# logsumexp para cada tamanho subtoken usando exemplos random 
def logsumexp_random_logZ(df_data, len_list, pattern_list, df_random, fill_number = 0):
    df = df_data.copy()
    df_r = df_random.copy()
    log_store = {}
    logsumexp_store = {}
    normalization = {}
    for size in len_list:
        log_store[size] = {}
        logsumexp_store[size] = {}
        for p in pattern_list:
            if p in log_store:
                raise ValueError
            values = df[(df.pattern == p) & (df.len_total == size)]
            values = values['bert_soma_total'].tolist()
            values_random = df_r[(df_r.pattern == p) & (df_r.len_total == size)]
            values_random = values_random['bert_soma_total'].tolist()
            random.shuffle(values_random)
            idx = fill_number-len(values) if fill_number > len(values) else 0
            values_random = values_random[:idx]
            values = values + values_random
            log_store[size][p] = torch.tensor(values)
            logsumexp_store[size][p] = torch.logsumexp(log_store[size][p], dim=0)

    df['log(Z)'] = df.apply(lambda row: logsumexp_store[row['len_total']][row['pattern']].item(), axis=1)
    # df['sum_bert_by_tokensize'] = df.apply(lambda row: normalization[row['len_total']][row['pattern']].item(), axis=1)
    # score final soma_total - log(Z)
    df['score_final_log(z)'] = df['bert_soma_total'] - df['log(Z)']
    # df['score_final_norm'] = df['bert_soma_total'] / df['sum_bert_by_tokensize']
    return df


def compute_dataframe_AP_by_pattern(df, key_sort, pattern_list):
    ap_by_pattern = {}
    for p in pattern_list:
        prec_list = []
        df_sorted = df[df['pattern'] == p]
        df_sorted = df_sorted.sort_values(by=key_sort, ascending=False)
        hyper_num = 0
        total_pair = 0
        for row in df_sorted.itertuples():
            total_pair += 1
            if row.fonte == 'hyper':
                hyper_num += 1
                prec_list.append(hyper_num / float(total_pair))

        ap_by_pattern[p] = np.mean(prec_list)
    return pd.DataFrame(data={'padrao': pattern_list, 'AP': list(ap_by_pattern.values())})


# compute ap in sorted list
def compute_AP(sorted_list):
    prec_list = []
    hyper_num = 0
    total_pair = 0
    for row in sorted_list:
        total_pair += 1
        hyper = row[0].strip().split()[3]
        if hyper == 'hyper':
            hyper_num += 1
            prec_list.append(hyper_num / float(total_pair))
    return np.mean(prec_list)


def compute_AP_by_rank(df, key_sort, best_patterns):
    rank = {}
    # compute rank
    for p in best_patterns:
        df_sorted = df[df['pattern'] == p]
        df_sorted = df_sorted.sort_values(by=key_sort, ascending=False, ignore_index=True)
        for row in df_sorted.itertuples():
            name = f"{row.hiponimo} {row.hiperonimo} {row.classe} {row.fonte}"
            if name in rank:
                rank[name].append(row.Index)
            else:
                rank[name] = []
                rank[name].append(row.Index)

    mean_rank = {}
    for row in rank:
        if row in mean_rank:
            raise ValueError
        mean_rank[row] = np.mean(rank[row])

    min_rank = {}
    for row in rank:
        if row in min_rank:
            raise ValueError
        min_rank[row] = min(rank[row])
    del rank
    sort_mean_rank = sorted(mean_rank.items(), key=lambda x: x[1])
    sort_min_rank = sorted(min_rank.items(), key=lambda x: x[1])
    del mean_rank, min_rank
    mean_ap = compute_AP(sort_mean_rank)
    min_ap = compute_AP(sort_min_rank)

    return min_ap, mean_ap


def compute_AP_n_best_pattern(df, key_sort, n_best_pattern):
    dict_values = {'n_best_pattern': [], 'method': [], 'AP': []}
    for num_p in range(1, len(n_best_pattern) + 1):
        min_ap, mean_ap = compute_AP_by_rank(df=df, key_sort=key_sort, best_patterns=n_best_pattern[:num_p])
        dict_values['n_best_pattern'].append(num_p)
        dict_values['method'].append('min ' + key_sort)
        dict_values['AP'].append(min_ap)
        dict_values['n_best_pattern'].append(num_p)
        dict_values['method'].append('mean ' + key_sort)
        dict_values['AP'].append(mean_ap)
    return pd.DataFrame(dict_values)


# devolve df de balanceamento True/False em todoo dataset
def balanceamento_all(df, patterns):
    df_rate = df[df['pattern'] == patterns[0]][['hiponimo', 'hiperonimo', 'classe', 'fonte', 'len_total']]
    df_rate = df_rate['fonte'].value_counts()
    hyper_num = df_rate['hyper']
    total = df_rate.sum()
    return pd.DataFrame(
        {'true': [hyper_num, (hyper_num / total)], 'false': [total - hyper_num, (total - hyper_num) / total],
         'total': [total, 1]})


def compute_min_mean_ap_normal(df_value, pattern_list, dataset_name, best_pattern_num=4):
    dfs = []
    method_score = ["score_final_log(z)", "score_final_norm"]

    for score_name in method_score[:1]:
        n_pair = df_value.groupby('pattern').count().iloc[0]['hiponimo']
        hyper_num = df_value[df_value['pattern'] == pattern_list[0]]['fonte'].value_counts()
        hyper_num = hyper_num['hyper']
        if score_name == "score_final_log(z)":
            min_ap, mean_ap = compute_AP_by_rank(df_value, key_sort=score_name,
                                                 best_patterns=pattern_list[:best_pattern_num])
        else:
            raise ValueError
        df = pd.DataFrame(
            {'dataset': [dataset_name] * 2, 'N': [n_pair] * 2, 'hyper_num': [hyper_num] * 2,
             'method': [f"min {score_name}", f"mean {score_name}"], 'AP': [min_ap, mean_ap]})

    # df_all = pd.concat([df, df_dive_word2vec])
    df_all = df
    df_all['method_format'] = df_all['method'].map(method_names)
    datasetnames_unique = df_all['dataset'].unique().tolist()
    rename_dataset = {}
    for k in datasetnames_unique:
        rename_dataset[k] = os.path.basename(k)

    df_all['dataset'] = df_all['dataset'].map(rename_dataset)
    return df_all


def compute_min_mean_ap_sep(df_value, pattern_list, dataset_name, best_pattern_num=4):
    perm_pattern = list(map(list, itertools.permutations(pattern_list[:best_pattern_num], r=2)))
    perm_pattern_list = []
    for i in perm_pattern:
        perm_pattern_list.append("_".join(i))

    dfs = []
    method_score = ["score_final_log(z)", "score_final_norm"]

    for score_name in method_score[:1]:
        n_pair = df_value.groupby('pattern').count().iloc[0]['hiponimo']
        hyper_num = df_value[df_value['pattern'] == perm_pattern_list[0]]['fonte'].value_counts()
        hyper_num = hyper_num['hyper']
        if score_name == "score_final_log(z)":
            min_ap, mean_ap = compute_AP_by_rank(df_value, key_sort=score_name,
                                                 best_patterns=perm_pattern_list)
        else:
            raise ValueError
        df = pd.DataFrame(
            {'dataset': [dataset_name] * 2, 'N': [n_pair] * 2, 'hyper_num': [hyper_num] * 2,
             'method': [f"min {score_name}", f"mean {score_name}"], 'AP': [min_ap, mean_ap]})

    # df_all = pd.concat([df, df_dive_word2vec])
    df_all = df
    df_all['method_format'] = df_all['method'].map(method_names)
    datasetnames_unique = df_all['dataset'].unique().tolist()
    rename_dataset = {}
    for k in datasetnames_unique:
        rename_dataset[k] = os.path.basename(k)

    df_all['dataset'] = df_all['dataset'].map(rename_dataset)
    return df_all


def compute_min_mean_ap_dot(df_value, pattern_list, dataset_name, best_pattern_num=4):
    perm_pattern = []
    pattern = pattern_list[:best_pattern_num]
    for i in range(2, len(pattern) + 1):
        tmp_p = list(map(list, itertools.permutations(pattern, r=i)))
        perm_pattern.extend(tmp_p)
    perm_pattern_list = []
    for i in perm_pattern:
        perm_pattern_list.append("_".join(i))

    dfs = []
    method_score = ["score_final_log(z)", "score_final_norm"]

    for score_name in method_score[:1]:
        n_pair = df_value.groupby('pattern').count().iloc[0]['hiponimo']
        hyper_num = df_value[df_value['pattern'] == perm_pattern_list[0]]['fonte'].value_counts()
        hyper_num = hyper_num['hyper']
        if score_name == "score_final_log(z)":
            min_ap, mean_ap = compute_AP_by_rank(df_value, key_sort=score_name,
                                                 best_patterns=perm_pattern_list)
        else:
            raise ValueError
        df = pd.DataFrame(
            {'dataset': [dataset_name] * 2, 'N': [n_pair] * 2, 'hyper_num': [hyper_num] * 2,
             'method': [f"min {score_name}", f"mean {score_name}"], 'AP': [min_ap, mean_ap]})

    # df_all = pd.concat([df, df_dive_word2vec])
    df_all = df
    df_all['method_format'] = df_all['method'].map(method_names)
    datasetnames_unique = df_all['dataset'].unique().tolist()
    rename_dataset = {}
    for k in datasetnames_unique:
        rename_dataset[k] = os.path.basename(k)

    df_all['dataset'] = df_all['dataset'].map(rename_dataset)
    return df_all


def compute_ap_bert_soma(df_value, pattern_list, dataset_name, tipo, best_pattern_num=4):
    if tipo == 'dot':
        perm_pattern = []
        pattern = pattern_list[:best_pattern_num]
        for i in range(2, len(pattern) + 1):
            tmp_p = list(map(list, itertools.permutations(pattern, r=i)))
            perm_pattern.extend(tmp_p)
        patterns = []
        for i in perm_pattern:
            patterns.append("_".join(i))
    elif tipo == 'sep':
        perm_pattern = list(map(list, itertools.permutations(pattern_list[:best_pattern_num], r=2)))
        patterns = []
        for i in perm_pattern:
            patterns.append("_".join(i))
    elif tipo == 'normal':
        patterns = pattern_list[:best_pattern_num]
    else:
        raise KeyError
    n_pair = df_value.groupby('pattern').count().iloc[0]['hiponimo']
    hyper_num = df_value[df_value['pattern'] == patterns[0]]['fonte'].value_counts()
    hyper_num = hyper_num['hyper']
    df = df_value[df_value['pattern'].isin(patterns)]
    min_ap, mean_ap = compute_AP_by_rank(df, key_sort='bert_soma_total',
                                         best_patterns=patterns)

    df = pd.DataFrame(
        {'dataset': [dataset_name] * 2, 'N': [n_pair] * 2, 'hyper_num': [hyper_num] * 2,
         'method': ["all_subword min_positional_rank", "all_subword mean_positional_rank"], 'AP': [min_ap, mean_ap]})

    df_all = df
    df_all['method_format'] = df_all['method'].map(method_names)
    datasetnames_unique = df_all['dataset'].unique().tolist()
    rename_dataset = {}
    for k in datasetnames_unique:
        rename_dataset[k] = os.path.basename(k)

    df_all['dataset'] = df_all['dataset'].map(rename_dataset)
    return df_all


def get_df_dive():
    algo = ['word2vec', 'summation_dot_product']
    dset_dive = {'baroni2012.json': [algo, [0.7176, 0.8344]], 'BLESS.json': [algo, [0.0911, 0.1552]],
                 'EVALution.json': [algo, [0.2546, 0.3415]], 'HypeNet_test.json': [algo, [0.2559, 0.3731]],
                 'kotlerman2010.json': [algo, [0.3950, 0.3659]], 'LenciBenotto.json': [algo, [0.4178, 0.5286]],
                 'levy2014.json': [algo, [0.1124, 0.1924]], 'turney2014.json': [algo, [0.5132, 0.5683]],
                 'Weeds.json': [algo, [0.5232, 0.6975]], 'wordnet_test.json': [algo, [0.5683, 0.5725]]
                 }
    dfs = []
    for dataset_name, values in dset_dive.items():
        df = pd.DataFrame({'dataset': [dataset_name] * 2, 'method': values[0], 'AP': values[1]})
        dfs.append(df)
    return pd.concat(dfs)


def get_dataset_names():
    dnames = {"kotlerman2010.json": "Kotlerman", "levy2014.json": "Medical", "turney2014.json": "TM 14",
              "baroni2012.json": "LEDS", "EVALution.json": "EVALution", "LenciBenotto.json": "LenciBenotto",
              "Weeds.json": "Weeds", "BLESS.json": "BLESS", "wordnet_test.json": "WordNet",
              "HypeNet_test.json": "HypeNet"
              }
    return dnames


def get_method_name_ijcai():
    mnames = {'word2vec': 'Word2vec C', 'summation_dot_product': 'DIVE \u0394S * C ', 'dot_product': 'DIVE C',
              'rnd': 'random', 'summation': 'DIVE \u0394S', 'summation_word2vec': 'DIVE \u0394S * Word2vec C',
              'all_subword mean_positional_rank': 'BERT Mean Pos Rank', 'all_subword min_positional_rank': 'BERT '
                                                                                                           'Min Pos '
                                                                                                           'Rank',
              'all_subword max_pattern': 'BERT Max Pattern', 'all_subword mean_pattern': 'BERT Mean Pattern',
              'min score_final_log(z)': 'BERT Min Rank log(Z)', 'min score_final_norm': 'BERT Min Pos Rank (/ '
                                                                                              'norm)',
              'mean score_final_log(z)': 'BERT Average Rank log(Z)', 'mean score_final_norm': 'BERT Mean Pos '
                                                                                                 'Rank (/ norm)',
              'min bert_soma_total': 'BERT Min Rank', 'mean bert_soma_total': 'BERT Average Rank',
              'BERT Min Pos Rank (log(z))': 'BERT Min Rank log(Z)' , 'BERT Mean Pos Rank (log(z))': 'BERT Average Rank log(Z)', 
              'BERT Min Pos Rank': 'BERT Min Rank', 'BERT Mean Pos Rank': 'BERT Average Rank'}
    return mnames
