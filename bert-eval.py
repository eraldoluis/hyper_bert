import numpy as np
import argparse
import random
import json
import os

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


def output2(dict_pairs, dataset_name, model_name, hyper_num, oov_num, f_out, patterns, include_oov = True):
    """

    :param dict_pairs: {'a b True hyper': { 'pattern1' : 0.1,
                                            'pattern2' : 0.2
                                            }
                        }
    """
    pair_position = {}
    for pattern in patterns:
        order_result = sorted(dict_pairs.items(), key=lambda x: x[1][pattern], reverse=True)
        for position, pair in enumerate(order_result):
            if pair[0] in pair_position:
                pair_position[pair[0]].append(position)
            else:
                pair_position[pair[0]] = []
                pair_position[pair[0]].append(position)

    order_final = sorted(pair_position.items(), key=lambda x: np.mean(x[1]), reverse=False)
    ap = compute_AP(order_final)
    f_out.write(f'{model_name}\t{dataset_name}\t{len(order_result)}\t{oov_num}\t{hyper_num}\t{"mean positional rank"}\t'
                f'{ap}\t{include_oov}\n')

    order_final2 = sorted(pair_position.items(), key=lambda x: min(x[1]), reverse=False)
    ap2 = compute_AP(order_final2)
    f_out.write(f'{model_name}\t{dataset_name}\t{len(order_result)}\t{oov_num}\t{hyper_num}\t{"min positional rank"}\t'
                f'{ap2}\t{include_oov}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_bert", type=str, help="path to json directory", required=True)
    parser.add_argument("-o", "--output_path", type=str, help="path to dir output", required=True)
    args = parser.parse_args()

    patterns = ["{} é um tipo de {}", "{} é um {}", "{} e outros {}", "{} ou outro {}", "{} , um {}"]


    # f_out.write(f'{model_name}\t{dataset_name}\t{len(order_result)}\t{oov_num}\t{hyper_num}\t{"mean positional rank"}\t'
    #             f'{ap}\t{include_oov}\n')
    try:
        os.mkdir(os.path.join(args.output_path, os.path.basename(args.input_bert)))
    except:
        pass
    f_out = open(os.path.join(args.output_path, os.path.basename(args.input_bert), "result.tsv"), mode="a")
    f_out.write("model\tdataset\tN\toov\thyper_num\tmethod\tAP\tinclude_oov\n")

    for filename in os.listdir(args.input_bert):
        if os.path.isfile(os.path.join(args.input_bert, filename)) and filename[-4:] == "json":
            with open(os.path.join(args.input_bert, filename)) as f_in:
                result = json.load(f_in)
                output2(result, filename, os.path.basename(args.input_bert), 1000, 1000, f_out, patterns, True)

    pass


if __name__ == '__main__':
    main()