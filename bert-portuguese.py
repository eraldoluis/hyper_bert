from transformers import BertConfig, BertForMaskedLM, BertTokenizer, BertModel
import torch
import torch.nn.functional as f
import logging
import numpy as np
import argparse
import random
import json
import os
random.seed(61)
logger = logging.getLogger(__name__)


class ClozeBert:
    def __init__(self, model_name, oov = True):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
        self.include_oov = oov

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.config = BertConfig.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=model_name.endswith("-uncased"))

        # self.tokenizer = BertTokenizer.from_pretrained(model_name + "/vocab.txt", do_lower_case=False)
        # self.models = BertForMaskedLM.from_pretrained(model_name, config=self.config)
        self.model = BertForMaskedLM.from_pretrained(model_name, config=self.config)
        self.model.to(self.device)


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
                examples = torch.tensor([example], device=self.device)
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

    def old_sentence_score(self, patterns, dataset, vocab_dive):
        words_probs_s = {}
        hyper = True
        oov = 0
        hyper_num = 0

        for row in dataset:
            pair = row[0:2]
            # if (pair[0] not in vocab_dive or pair[1] not in vocab_dive) and (args.include_oov):
            #     # par nao está no vocab do dive e calculo deverá incluí-lo
            #     words_probs_s[" ".join(row)] = {}
            #     words_probs_s[" ".join(row)]['oov'] = float("-inf")
            #     oov += 1
            #     if row[3] == "hyper":
            #         hyper_num += 1
            #     continue

            if (not self.include_oov) and (pair[0] not in vocab_dive or pair[1] not in vocab_dive):
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
                        examples = torch.tensor([tokenized_text], device=self.device)
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

    def sentence_score(self, patterns, dataset, vocab_dive):
        words_probs_s = {}
        hyper = True
        oov = 0
        hyper_num = 0

        for row in dataset:
            pair = row[0:2]
            # if (pair[0] not in vocab_dive or pair[1] not in vocab_dive) and (args.include_oov):
            #     # par nao está no vocab do dive e calculo deverá incluí-lo
            #     words_probs_s[" ".join(row)] = {}
            #     words_probs_s[" ".join(row)]['oov'] = float("-inf")
            #     oov += 1
            #     if row[3] == "hyper":
            #         hyper_num += 1
            #     continue

            if (not self.include_oov) and (pair[0] not in vocab_dive or pair[1] not in vocab_dive):
                oov += 1
                # par nao está no vocab do dive e calculo NÃO deverá incluí-lo
                continue

            words_probs_s[" ".join(row)] = {}
            if row[3] == "hyper":
                hyper_num += 1

            for pattern in patterns:
                sentences, idx_h, idx_mask = self.build_sentences(pattern, pair)
                idx_all = idx_h[0].copy()
                idx_all.extend(idx_h[1])
                hyponym_idx, hypernym_idx = idx_h[0], idx_h[1]
                sentences_hyponym = sentences[:len(hyponym_idx)]
                sentences_hypernym = sentences[-len(hypernym_idx):]
                logger.info("Predicting...")
                self.model.eval()
                with torch.no_grad():
                    examples = torch.tensor(sentences, device=self.device)
                    # segments_tensors = torch.tensor([segments_ids])
                    outputs = self.model(examples)  # , segments_tensors)
                predict = outputs[0]
                predict = f.log_softmax(predict, dim=2)
                predict = predict[torch.arange(len(sentences), device=self.device), idx_mask, idx_all]

                # predict = torch.diagonal(predict[:, idx_mask, idx_all], 0)

                predict_hypon = predict[:len(idx_h[0])]
                # print(predict_hypon)
                predict_hyper = predict[- len(idx_h[1]):]
                # print(predict_hyper)
                # predict for sentences. shape( len(sentences) )
                # print(predict)
                words_probs_s[" ".join(row)][pattern] = []
                words_probs_s[" ".join(row)][pattern].append(predict_hypon.numpy().tolist())
                words_probs_s[" ".join(row)][pattern].append(predict_hyper.numpy().tolist())
                # words_probs_s[" ".join(row)][pattern] = torch.sum(predict).item()

        return words_probs_s, hyper_num, oov

    def build_sentences(self, pattern, pair):  # feito, agora falta tratar onde isso eh chamado
        sentence1 = pattern.format("[MASK]", pair[1])
        sentence2 = pattern.format(pair[0], "[MASK]")
        logger.info("Tokenizing...")
        hyponym_tokenize = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(pair[0]))
        hypernym_tokenize = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(pair[1]))
        pattern1_tokenize = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sentence1))
        pattern2_tokenize = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sentence2))

        data = []
        sentences = []
        idx_masks = []
        # hyponym
        # if len(hyponym_tokenize) > 1: # sub-word
        idx_mask = pattern1_tokenize.index(self.tokenizer.mask_token_id)
        antes = pattern1_tokenize[:idx_mask]
        depois = pattern1_tokenize[idx_mask + 1:]
        for i in range(len(hyponym_tokenize)):
            sentence = hyponym_tokenize.copy()
            sentence[i] = self.tokenizer.mask_token_id
            data.append(sentence)

        for i in range(len(data)):
            row = antes.copy()
            row.extend(data[i])
            row.extend(depois)
            idx_masks.append(row.index(self.tokenizer.mask_token_id))
            sentences.append(row)

        # hypernym
        data = []
        idx_mask = pattern2_tokenize.index(self.tokenizer.mask_token_id)
        antes = pattern2_tokenize[:idx_mask]
        depois = pattern2_tokenize[idx_mask + 1:]
        for i in range(len(hypernym_tokenize)):
            sentence = hypernym_tokenize.copy()
            sentence[i] = self.tokenizer.mask_token_id
            data.append(sentence)

        for i in range(len(data)):
            row = antes.copy()
            row.extend(data[i])
            row.extend(depois)
            idx_masks.append(row.index(self.tokenizer.mask_token_id))
            sentences.append(row)

        ids = [hyponym_tokenize, hypernym_tokenize]
        return sentences, ids, idx_masks


def load_eval_file(f_in):
    eval_data = []
    for line in f_in:
        child_pos, parent_pos, is_hyper, rel = line.strip().split('\t')
        child = child_pos.strip()
        parent = parent_pos.strip()
        is_hyper = is_hyper.strip()
        rel = rel.strip()
        eval_data.append([child, parent, is_hyper, rel])
    random.shuffle(eval_data)
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

def save_bert_file(dict, output, dataset_name, model_name, hyper_num, oov_num, f_info_out, include_oov = True):
    logger.info("save info...")
    f_info_out.write(f'{model_name}\t{dataset_name}\t{len(dict)}\t{oov_num}\t{hyper_num}\t{include_oov}\n')
    logger.info("save json...")
    dname = os.path.splitext(dataset_name)[0]
    fjson = json.dumps(dict, ensure_ascii=False)
    f = open(os.path.join(output, model_name.replace("/","-"), dname + ".json"), mode="w", encoding="utf-8")
    f.write(fjson)
    f.close()

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


def output(dict_pairs, dataset_name, model_name, hyper_num, oov_num, f_out, include_oov=True):
    """

    :param dict_pairs: {'a b True hyper': { 'pattern1' : 0.1,
                                            'pattern2' : 0.2
                                            }
                        }
    """
    methods = ["sum", "prod", "max", "min"]
    result_by_par = {}
    for pair, patterns in dict_pairs.items():
        result_by_par[pair] = {}
        result_by_par[pair]['sum'] = sum(patterns.values())
        result_by_par[pair]['prod'] = np.prod(list(patterns.values()))
        result_by_par[pair]['max'] = max(patterns.values())
        result_by_par[pair]['min'] = min(patterns.values())
    for method in methods:
        order_result = sorted(result_by_par.items(), key=lambda x: x[1][method], reverse=False)
        ap = compute_AP(order_result)
        # models dataset N oov hyper_num method AP include_oov
        f_out.write(f'{model_name}\t{dataset_name}\t{len(order_result)}\t{oov_num}\t{hyper_num}\t{method}\t'
                    f'{ap}\t{include_oov}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, help="path to bert models", required=True)
    parser.add_argument("-e", "--eval_path", type=str, help="path to datasets", required=True)
    parser.add_argument("-o", "--output_path", type=str, help="path to dir output", required=False)
    parser.add_argument("-v", "--vocab", type=str, help="dir of vocab", required=False)
    parser.add_argument("-u", "--include_oov", action="store_true", help="to include oov on results")

    args = parser.parse_args()
    print("Iniciando bert...")
    cloze_model = ClozeBert(args.model_name)
    try:
        os.mkdir(os.path.join(args.output_path, args.model_name.replace("/", "-")))
    except:
        pass

    f_out = open(os.path.join(args.output_path, args.model_name.replace('/', '-'), "info.tsv"), mode="a")
    f_out.write("model\tdataset\tN\toov\thyper_num\tinclude_oov\n")

    patterns = ["{} é um tipo de {}", "{} é um {}", "{} e outros {}", "{} ou outro {}", "{} , um {}"]
    # patterns = ["[MASK] é um tipo de [MASK]", "[MASK] é um [MASK]"]

    pairs = [['tigre', 'animal', 'True', 'hyper'], ['casa', 'moradia', 'True', 'hyper'],
             ['banana', 'abacate', 'False', 'random']]

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
                print(f"dataset={filedataset} size={len(eval_data)}")
                result, hyper_total, oov_num = cloze_model.sentence_score(patterns, eval_data[:10], dive_vocab)
                save_bert_file(result, args.output_path, filedataset, args.model_name, hyper_total, oov_num, f_out, args.include_oov)
                logger.info(f"result_size={len(result)}")
                # output2(result, filedataset, args.model_name, hyper_total, oov_num, f_out, patterns, args.include_oov)
    f_out.close()
    logger.info("Done")
    print("Done!")


if __name__ == "__main__":
    t = main()
