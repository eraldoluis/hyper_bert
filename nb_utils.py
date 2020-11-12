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
            print(f"Balanceamento: {v} não está no dataframe!")

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
        normalization[size] = {}
        for p in pattern_list:
            if p in log_store:
                raise ValueError
            values = df[(df.pattern == p) & (df.len_total == size)]
            log_store[size][p] = torch.tensor(values['bert_soma_total'].tolist())
            logsumexp_store[size][p] = torch.logsumexp(log_store[size][p], dim=0)
            normalization[size][p] = torch.sum(log_store[size][p])

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
