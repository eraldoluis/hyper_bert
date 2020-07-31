from transformers import BertConfig, BertForMaskedLM, BertTokenizer, BertModel
import torch
import logging
import numpy as np
import argparse
import random
import os

logger = logging.getLogger(__name__)


class ClozeBert:
    def __init__(self, model_name):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
        self.config = BertConfig.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name + "/vocab.txt", do_lower_case=False)
        self.model = BertForMaskedLM.from_pretrained(model_name, config=self.config)

        # self.model = BertModel.from_pretrained(model_name)

    def most_probabable_words(self, texts):
        words_probs_s = []
        for text in texts:
            logger.info("Tokenizing...")
            tokenized_text = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
            example = self.tokenizer.build_inputs_with_special_tokens(tokenized_text)

            idx_mask = example.index(self.tokenizer.mask_token_id)

            logger.info("Predicting...")
            self.model.eval()
            with torch.no_grad():
                examples = torch.tensor([example])
                outputs, = self.model(examples)

            # outputs shape is (batch_example, words, scores).
            probs_mask = outputs[0, idx_mask]

            logger.info("Zipping...")
            words_probs = zip(probs_mask, self.tokenizer.vocab.keys())

            logger.info("Sorting...")
            # pair words with their scores (score, word) and sort them by score
            words_probs = sorted(words_probs, reverse=True)

            words_probs_s.append(words_probs)

        return words_probs_s

    def sentence_score(self, patterns, dataset, vocab_dive):
        words_probs_s = {}
        hyper = True
        oov = 0
        hyper_num = 0

        for row in dataset:
            pair = row[0:2]
            if (pair[0] not in vocab_dive or pair[1] not in vocab_dive) and (args.include_oov):
                # par nao está no vocab do dive e calculo deverá incluí-lo
                words_probs_s[" ".join(row)] = {}
                words_probs_s[" ".join(row)]['oov'] = float("-inf")
                oov += 1
                if row[3] == "hyper":
                    hyper_num += 1
                continue

            if (pair[0] not in vocab_dive or pair[1] not in vocab_dive) and (args.include_oov == False):
                oov += 1
                # par nao está no vocab do dive e calculo NÃO deverá incluí-lo
                continue

            words_probs_s[" ".join(row)] = {}
            if row[3] == "hyper":
                hyper_num += 1

            tokenized_hipo = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(pair[0]))
            tokenized_hype = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(pair[1]))
            for pattern in patterns:
                pattern_prob = []
                string_pattern_list = self.build_sentences(pattern, pair)
                for str_pattern in string_pattern_list:
                    logger.info("Tokenizing...")
                    tokenized_text = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(str_pattern))

                    # example = self.tokenizer.build_inputs_with_special_tokens(tokenized_text) # initial and end token

                    # idx_mask = example.index(self.tokenizer.mask_token_id)
                    idx_mask = tokenized_text.index(self.tokenizer.mask_token_id)
                    # segments_ids = [0] * len(tokenized_text)

                    tokenized_mask = tokenized_hype if hyper else tokenized_hipo

                    logger.info("Predicting...")
                    self.model.eval()
                    with torch.no_grad():
                        examples = torch.tensor([tokenized_text])
                        # segments_tensors = torch.tensor([segments_ids])
                        outputs = self.model(examples)  # , segments_tensors)

                    # outputs shape is (batch_example, words, scores).
                    probs_mask = outputs[0][0, idx_mask, tokenized_mask]
                    hyper = not hyper
                    # probs_mask = torch.mean(probs_mask).item()
                    probs_mask = torch.prod(probs_mask).item()
                    pattern_prob.append(probs_mask)

                words_probs_s[" ".join(row)][pattern] = np.prod(pattern_prob)

        return words_probs_s, hyper_num, oov

    def build_sentences(self, pattern, pair):
        sentence1 = pattern.format(pair[0], "[MASK]")
        sentence2 = pattern.format("[MASK]", pair[1])
        return [sentence1, sentence2]


def load_eval_file(f_in):
    eval_data = []
    for line in f_in:
        line = line[:-1]
        # print line
        child_pos, parent_pos, is_hyper, rel = line.split('\t')
        child = child_pos.strip()
        parent = parent_pos.strip()
        is_hyper = is_hyper.strip()
        rel = rel.strip()
        eval_data.append([child, parent, is_hyper, rel])
    # random.shuffle(eval_data)
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


def output(dict_pairs, dataset_name, model_name, hyper_num, oov_num, f_out):
    """

    :param dict_pairs: {'a b True hyper': { 'pattern1' : 0.1,
                                            'pattern2' : 0.2
                                            }
                        }
    :param dataset_name: str
    """
    methods = ["sum", "prod", "max"]
    # somar o score dos padrões para cada par
    result_by_par = {}
    for pair, patterns in dict_pairs.items():
        result_by_par[pair] = {}
        result_by_par[pair]['sum'] = sum(patterns.values())
        result_by_par[pair]['prod'] = np.prod(list(patterns.values()))
        result_by_par[pair]['max'] = max(patterns.values())
    for method in methods:
        order_result = sorted(result_by_par.items(), key=lambda x: x[1][method], reverse=True)
        ap = compute_AP(order_result)
        # model dataset N oov hyper_num method AP include_oov
        f_out.write(f'{model_name}\t{dataset_name}\t{len(order_result)}\t{oov_num}\t{hyper_num}\t{method}\t'
                    f'{ap}\t{args.include_oov}\n')


def main():
    path_bert = "/home/gabriel/Downloads/bert-base-portuguese-cased"
    # cloze = ClozeBert(path_bert)
    cloze = ClozeBert(args.model_name)

    model_name = os.path.basename(os.path.normpath(args.model_name))
    try:
        os.mkdir(args.output_path)
    except:
        pass

    f_out = open(os.path.join(args.output_path, f"result_{model_name}" + ".tsv"), mode="w")
    f_out.write("model\tdataset\tN\toov\thyper_num\tmethod\tAP\tinclude_oov\n")

    patterns = ["{} é um tipo de {}", "{} é um {}"]

    logger.info("Loading vocab dive ...")
    dive_vocab = []
    with open(os.path.join(args.vocab, "vocab.txt"), mode="r", encoding="utf-8") as f_vocab:
        for line in f_vocab:
            word, count = line.strip().split()
            dive_vocab.append(word)

    for filedataset in os.listdir(args.eval_path):
        if os.path.isfile(os.path.join(args.eval_path, filedataset)):
            with open(os.path.join(args.eval_path, filedataset)) as f_in:
                logger.info("Loading dataset ...")
                eval_data = load_eval_file(f_in)
                result, hyper_total, oov_num = cloze.sentence_score(patterns, eval_data, dive_vocab)
                output(result, filedataset, model_name, hyper_total, oov_num, f_out)
    f_out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, help="path to bert model", required=True)
    parser.add_argument("-e", "--eval_path", type=str, help="path to datasets", required=True)
    parser.add_argument("-o", "--output_path", type=str, help="path to dir output", required=True)
    parser.add_argument("-v", "--vocab", type=str, help="dir of vocab", required=True)
    parser.add_argument("-u", "--include_oov", action="store_true", help="to include oov on results")

    args = parser.parse_args()
    main()
