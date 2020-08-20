import operator

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


def output2(dict_pairs, dataset_name, model_name, hyper_num, oov_num, f_out, patterns, include_oov=True):

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
                for pattern_name, values in pattern.items():
                    new_pairs[data][pattern_name] = sum(list(sum(values, [])))

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
                for pattern_name, values in pattern.items():
                    new_pairs[data][pattern_name] = np.mean(values[0]) + np.mean(values[1])

        else:
            raise ValueError

        for s_m in sub_method:
            if s_m == "mean_positional_rank" or s_m == "min_positional_rank":
                # faz um rank para cada pattern
                pair_position = {}
                for pattern in patterns:
                    order_result = sorted(new_pairs.items(), key=lambda x: x[1][pattern], reverse=True)
                    for position, pair in enumerate(order_result):
                        if pair[0] in pair_position:
                            pair_position[pair[0]].append(position)
                        else:
                            pair_position[pair[0]] = []
                            pair_position[pair[0]].append(position)
                if s_m ==  "mean_positional_rank":
                    # faz a media dos rankings
                    order_final = sorted(pair_position.items(), key=lambda x: np.mean(x[1]), reverse=False)
                    ap = compute_AP(order_final)
                    f_out.write(
                        f'{model_name}\t{dataset_name}\t{len(order_final)}\t{oov_num}\t{hyper_num}\t{m} {s_m}\t'
                        f'{ap}\t{include_oov}\n')
                elif s_m == "min_positional_rank":
                    # usa o menor ranking para cada par
                    order_final = sorted(pair_position.items(), key=lambda x: min(x[1]), reverse=False)
                    ap = compute_AP(order_final)
                    f_out.write(
                        f'{model_name}\t{dataset_name}\t{len(order_final)}\t{oov_num}\t{hyper_num}\t{m} {s_m}\t'
                        f'{ap}\t{include_oov}\n')
                else:
                    raise ValueError
            elif s_m == "max_pattern":
                max_pattern = {}
                for data, pattern in new_pairs.items():
                    max_pattern[data] = max(pattern.items(), key=operator.itemgetter(1))[1]
                order_final = sorted(max_pattern.items(), key=lambda x: x[1], reverse=True)

                ap = compute_AP(order_final)
                f_out.write(
                    f'{model_name}\t{dataset_name}\t{len(order_final)}\t{oov_num}\t{hyper_num}\t{m} {s_m}\t'
                    f'{ap}\t{include_oov}\n')
            elif s_m == "mean_pattern":
                mean_pattern = {}
                for data, patterns in new_pairs.items():
                    mean = []
                    for pattern, value in patterns.items():
                        mean.append(value)
                    mean_pattern[data] = np.mean(mean)

                order_final = sorted(mean_pattern.items(), key=lambda x: x[1], reverse=True)
                ap = compute_AP(order_final)
                f_out.write(
                    f'{model_name}\t{dataset_name}\t{len(order_final)}\t{oov_num}\t{hyper_num}\t{m} {s_m}\t'
                    f'{ap}\t{include_oov}\n')
            else:
                raise ValueError


def main():
    logger.info("Iniciando...")
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_bert", type=str, help="path to json directory", required=True)
    parser.add_argument("-o", "--output_path", type=str, help="path to dir output", required=True)
    args = parser.parse_args()

    patterns = ["{} é um tipo de {}", "{} é um {}", "{} e outros {}", "{} ou outro {}", "{} , um {}"]

    # f_out.write(f'{model_name}\t{dataset_name}\t{len(order_result)}\t{oov_num}\t{hyper_num}\t{"mean positional rank"}\t'
    #             f'{ap}\t{include_oov}\n')
    try:
        os.mkdir(os.path.join(args.output_path, os.path.basename(args.input_bert)))
    except ValueError:
        raise ValueError

    f_out = open(os.path.join(args.output_path, os.path.basename(args.input_bert), "result.tsv"), mode="a")
    f_out.write("model\tdataset\tN\toov\thyper_num\tmethod\tAP\tinclude_oov\n")

    logger.info("Carregando info.tsv")
    df_info = pd.read_csv(os.path.join(args.input_bert, "info.tsv"), delimiter="\t")
    for filename in os.listdir(args.input_bert):
        if os.path.isfile(os.path.join(args.input_bert, filename)) and filename[-4:] == "json":
            dataset_name = os.path.splitext(filename)[0] + ".tsv"
            n_size = \
            df_info[(df_info['model'] == os.path.basename(args.input_bert)) & (df_info['dataset'] == dataset_name)][
                'N'].squeeze()
            oov_num = \
            df_info[(df_info['model'] == os.path.basename(args.input_bert)) & (df_info['dataset'] == dataset_name)][
                'oov'].squeeze()
            hyper_num = \
            df_info[(df_info['model'] == os.path.basename(args.input_bert)) & (df_info['dataset'] == dataset_name)][
                'hyper_num'].squeeze()
            include_oov = \
            df_info[(df_info['model'] == os.path.basename(args.input_bert)) & (df_info['dataset'] == dataset_name)][
                'include_oov'].squeeze()

            with open(os.path.join(args.input_bert, filename)) as f_in:
                logger.info(f"Carregando json {filename}")
                result = json.load(f_in)
                output2(result, dataset_name, os.path.basename(args.input_bert), hyper_num, oov_num, f_out, patterns,
                        include_oov)
    f_out.close()
    logger.info("Done!")


if __name__ == '__main__':
    main()
