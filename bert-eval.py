import operator
import sys

import numpy as np
import argparse
import logging
import pandas as pd
import random
import json
import os

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

def load_eval_file(f_in):
    eval_data = []
    for line in f_in:
        line = line[:-1]
        eval_data.append(line.replace("\t", " "))
    return eval_data



def compute_AP(result_list_method):
    """

    :type result_list_method: [ ('a b True hyper',
                                {'sum': 0.1, 'prod': 0.4, 'max': 0.8}
                                ) ]
    """
    prec_list = []
    correct_count = 0
    all_count = 0

    total_sample_num = len(result_list_method)
    num_hyper = 0
    for i in range(total_sample_num):
        row = result_list_method[i][0]  # hipo hyper True hyper
        rel = row.split()[3]
        if rel == "hyper":
            num_hyper += 1

    for i in range(total_sample_num):
        all_count += 1
        row = result_list_method[i][0]  # hipo hyper True hyper
        rel = row.split()[3]
        if rel == "hyper":
            correct_count += 1
            precision = correct_count / float(all_count)
            prec_list.append(precision)

    return np.mean(prec_list)


def infos_eval(dict_result):
    hyper_num = 0
    for k, v in dict_result.items():
        row = k.strip().split()
        if row[3] == "hyper":
            hyper_num += 1
    return hyper_num


def output2(dict_pairs, dataset_name, model_name, f_out, patterns_list, corpus, include_oov=True):

    hyper_num = infos_eval(dict_pairs)
    oov_num = 0
    logger.info("Calculando score...")
    method = ["mean_subword", "all_subword"]
    sub_method = ["mean_positional_rank", "min_positional_rank", "max_pattern", "mean_pattern"]
    for m in method:
        if m == "all_subword":
            """

               :param new_pairs: {'a b True hyper': { 'pattern1' : 0.1,
                                                       'pattern2' : 0.2
                                                       }
                                   }
            """
            new_pairs = {}
            for data, pattern in dict_pairs.items():
                new_pairs[data] = {}
                for pattern_name in patterns_list:
                    values = dict_pairs[data][pattern_name]
                    if "z_score" in dict_pairs[data]:
                        raise ValueError
                        new_pairs[data][pattern_name] = (sum(list(sum(values, [])))) / dict_pairs[data]['z_score'][pattern_name]
                    else:
                        new_pairs[data][pattern_name] = (sum(list(sum(values, []))))

        elif m == "mean_subword":
            """
               em pattern 1 o hiponimo é quebrado em 2 subwords com scores=[-10,-20]                                                       
               :param dict_pairs: {'a b True hyper': { 'pattern1' : [[-10, -20], [-5]],
                                                       'pattern2' : 0.2
                                                       }
                                   }
            """
            new_pairs = {}
            for data, pattern in dict_pairs.items():
                new_pairs[data] = {}
                for pattern_name in patterns_list:
                    values = dict_pairs[data][pattern_name]
                    if "z_score" in dict_pairs[data]:
                        raise ValueError
                        new_pairs[data][pattern_name] = (np.mean(values[0]) + np.mean(values[1])) / \
                                                        dict_pairs[data]['z_score'][pattern_name]
                    else:
                        new_pairs[data][pattern_name] = (np.mean(values[0]) + np.mean(values[1]))

        else:
            raise ValueError

        for s_m in sub_method:
            if s_m == "mean_positional_rank" or s_m == "min_positional_rank":
                # faz um rank para cada pattern
                pair_position = {}
                for pattern_name in patterns_list:
                    order_result = sorted(new_pairs.items(), key=lambda x: x[1][pattern_name], reverse=True)
                    for position, pair in enumerate(order_result):
                        if pair[0] in pair_position:
                            pair_position[pair[0]].append(position)
                        else:
                            pair_position[pair[0]] = []
                            pair_position[pair[0]].append(position)
                if s_m == "mean_positional_rank":
                    # faz a media dos rankings
                    order_final = sorted(pair_position.items(), key=lambda x: np.mean(x[1]), reverse=False)
                    ap = compute_AP(order_final)
                elif s_m == "min_positional_rank":
                    # usa o menor ranking para cada par
                    order_final = sorted(pair_position.items(), key=lambda x: min(x[1]), reverse=False)
                    ap = compute_AP(order_final)
                else:
                    raise ValueError
            elif s_m == "max_pattern":
                max_pattern = {}
                for data, pattern_name in new_pairs.items():
                    max_pattern[data] = max(pattern_name.items(), key=operator.itemgetter(1))[1]
                order_final = sorted(max_pattern.items(), key=lambda x: x[1], reverse=True)

                ap = compute_AP(order_final)
            elif s_m == "mean_pattern":
                mean_pattern = {}
                for data, patterns in new_pairs.items():
                    mean = []
                    for pattern_name, value in patterns.items():
                        mean.append(value)
                    mean_pattern[data] = np.mean(mean)

                order_final = sorted(mean_pattern.items(), key=lambda x: x[1], reverse=True)
                ap = compute_AP(order_final)
            else:
                raise ValueError

            f_out.write(
                f'{model_name}\t{dataset_name}\t{len(order_final)}\t{oov_num}\t{hyper_num}\t{m} {s_m}\t'
                f'{ap}\t{include_oov}\t{corpus}\t{len(patterns_list)}\n')


def output_by_pattern(dict_pairs, dataset_name, model_name, f_out, patterns_list, corpus, include_oov=True):
    hyper_num = infos_eval(dict_pairs)
    oov_num = 0
    logger.info("Calculando score...")
    method = ["mean_subword", "all_subword"]
    sub_method = ["mean_positional_rank", "min_positional_rank", "max_pattern", "mean_pattern"]
    for m in method:
        if m == "all_subword":
            """

               :param new_pairs: {'a b True hyper': { 'pattern1' : 0.1,
                                                       'pattern2' : 0.2
                                                       }
                                   }
            """
            new_pairs = {}
            for data, pattern in dict_pairs.items():
                new_pairs[data] = {}
                for pattern_name in patterns_list:
                    values = dict_pairs[data][pattern_name]
                    if len(values[0]) > 1 or len(values[1]) > 1:
                        raise ValueError
                    if "z_score" in dict_pairs[data]:
                        new_pairs[data][pattern_name] = (sum(list(sum(values, [])))) / dict_pairs[data]['z_score'][pattern_name]
                    else:
                        new_pairs[data][pattern_name] = (sum(list(sum(values, []))))
        elif m == "mean_subword":
            """
               em pattern 1 o hiponimo é quebrado em 2 subwords com scores=[-10,-20]                                                       
               :param dict_pairs: {'a b True hyper': { 'pattern1' : [[-10, -20], [-5]],
                                                       'pattern2' : 0.2
                                                       }
                                   }
            """
            new_pairs = {}
            for data, pattern in dict_pairs.items():
                new_pairs[data] = {}
                for pattern_name in patterns_list:
                    values = dict_pairs[data][pattern_name]
                    if len(values[0]) > 1 or len(values[1]) > 1:
                        raise ValueError
                    if "z_score" in dict_pairs[data]:
                        new_pairs[data][pattern_name] = (np.mean(values[0]) + np.mean(values[1])) / dict_pairs[data]['z_score'][pattern_name]
                    else:
                        new_pairs[data][pattern_name] = (np.mean(values[0]) + np.mean(values[1]))
        else:
            raise ValueError

        for pattern_name in patterns_list:
            order_result = sorted(new_pairs.items(), key=lambda x: x[1][pattern_name], reverse=True)
            ap = compute_AP(order_result)
            f_out.write(
                f'{model_name}\t{dataset_name}\t{len(order_result)}\t{oov_num}\t{hyper_num}\t{m}\t'
                f'{ap}\t{include_oov}\t{corpus}\t{pattern_name}\n')


def read_vocab(path_vocab):
    vocab = []
    with open(path_vocab, mode="r", encoding="utf-8") as f:
        for line in f:
            w, c = line.strip().split()
            vocab.append(w)
    corpus = path_vocab.split("/")[-2]
    return vocab, corpus


def filter_oov(data, vocab):
    new_data = {}
    for k, v in data.items():
        row = k.split()
        if row[0] in vocab and row[1] in vocab:
            new_data[k] = v
    return  new_data


def main():
    logger.info("Iniciando...")
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_bert", type=str, help="path to json directory", required=True)
    parser.add_argument("-o", "--output_path", type=str, help="dir output", required=True)
    parser.add_argument("-e", "--eval_path", type=str, help="dir datasets", required=True)
    parser.add_argument("--vocabs", type=str, help="dir vocabs", required=False)
    args = parser.parse_args()

    patterns = ["{} é um tipo de {}", "{} é um {}", "{} e outros {}", "{} ou outro {}", "{} , um {}"]
    patterns2 = ["{} que é um exemplo de {}", "{} que é uma classe de {}", "{} que é um tipo de {}",
                 "{} e qualquer outro {}", "{} e algum outro {}", "{} ou qualquer outro {}", "{} ou algum outro {}",
                 "{} que é chamado de {}",
                 "{} é um caso especial de {}",
                 "{} incluindo {}"]
    patterns.extend(patterns2)

    best_bert_score = ['{} que é um exemplo de {}', '{} incluindo {}', '{} que é chamado de {}', '{} é um tipo de {}',
                       '{} é um {}', '{} e outros {}', '{} que é um tipo de {}', '{} é um caso especial de {}',
                       '{} que é uma classe de {}', '{} e qualquer outro {}', '{} ou qualquer outro {}', '{} , um {}',
                       '{} ou outro {}', '{} ou algum outro {}', '{} e algum outro {}']

    assert len(patterns) == len(best_bert_score) and set(patterns) == set(best_bert_score)

    # f_out.write(f'{model_name}\t{dataset_name}\t{len(order_result)}\t{oov_num}\t{hyper_num}\t{"mean positional rank"}\t'
    #             f'{ap}\t{include_oov}\n')
    try:
        dir = os.path.join(args.output_path, os.path.basename(args.input_bert)+"_sort-best-pattern")
        os.mkdir(dir)
    except ValueError:
        raise ValueError

    f_out = open(os.path.join(dir, "result.tsv"), mode="a")
    f_out.write("model\tdataset\tN\toov\thyper_num\tmethod\tAP\tinclude_oov\tcorpus\tqts_pattern\n")

    # f_out.write("model\tdataset\tN\toov\thyper_num\tmethod\tAP\tinclude_oov\tcorpus\tpattern\tqts_pattern\n")

    logger.info("Carregando datasets")
    dataset = {}
    for filename in os.listdir(args.eval_path):
        if os.path.isfile(os.path.join(args.eval_path, filename)):
            with open(os.path.join(args.eval_path, filename), mode="r", encoding="utf-8") as f:
                data = load_eval_file(f)
                dataset_name_token1 = filename
                dataset[dataset_name_token1] = data

    for filename in os.listdir(args.input_bert):
        logger.info(f"file={filename}\t{dataset_name_token1}")
        if os.path.isfile(os.path.join(args.input_bert, filename)) and filename[-4:] == "json" and os.path.splitext(filename)[0] + ".tsv" in dataset:
            dataset_name = os.path.splitext(filename)[0] + ".tsv"

            with open(os.path.join(args.input_bert, filename)) as f_in:
                logger.info(f"Carregando json {filename}")
                result = json.load(f_in)
                new_result = {}
                # filtrando conforme o novo dataset de subtoken de tamanho 1
                for i in dataset[dataset_name]:
                    if i in result:
                        new_result[i] = result[i]

                #filtrando oov conforme vocab dive
                for v_p in os.listdir(args.vocabs):
                    vocab, corpus_name = read_vocab(os.path.join(args.vocabs, v_p, "vocab.txt"))
                    dict_result = filter_oov(new_result, vocab)
                    logger.info("filtrando datasets")
                    for qtd_best_pattern in range(1, len(best_bert_score)+1):
                        output2(dict_result, dataset_name, os.path.basename(args.input_bert), f_out, best_bert_score[:qtd_best_pattern],
                                corpus_name, args.vocabs is None)
                    # output_by_pattern(dict_result, dataset_name, os.path.basename(args.input_bert), f_out, patterns, corpus_name,
                    #         args.vocabs is None)
                # output2(new_result, dataset_name, os.path.basename(args.input_bert), f_out, patterns, "bert",
                #     not args.vocabs is None)
                for qtd_best_pattern in range(1, len(best_bert_score) + 1):
                    output2(new_result, dataset_name, os.path.basename(args.input_bert), f_out,
                            best_bert_score[:qtd_best_pattern], "bert", not args.vocabs is None)
                # output_by_pattern(new_result, dataset_name, os.path.basename(args.input_bert), f_out, patterns, "bert",
                #         not args.vocabs is None)
    f_out.close()
    logger.info("Done!")


if __name__ == '__main__':
    main()
