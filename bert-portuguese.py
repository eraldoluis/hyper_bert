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

    def sentence_score(self, patterns, dataset):
        words_probs_s = {}
        hyper = True
        for row in dataset:
            pair = row[0:2]
            words_probs_s[" ".join(row)] = {}
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

        return words_probs_s

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
    random.shuffle(eval_data)
    return eval_data


def main():
    path_bert = "/home/gabriel/Downloads/bert-base-portuguese-cased"
    # cloze = ClozeBert(path_bert)
    cloze = ClozeBert(args.model_name)

    patterns = ["{} é um tipo de {}"]
    # pairs = [['tigre', 'animal', 'true', 'hyper'], ['casa', 'construção', 'true', 'hyper'],
    #          ['joelho', 'carro', 'false', 'random']]
    # result = cloze.sentence_score(patterns, pairs)

    for filedataset in os.listdir(args.eval_path):
        if os.path.isfile(os.path.join(args.eval_path, filedataset)):
            with open(os.path.join(args.eval_path, filedataset)) as f_in:
                eval_data = load_eval_file(f_in)
                result = cloze.sentence_score(patterns, eval_data[:10])
                # somar o score dos métodos
                result_by_par = {}
                for pair, patterns in result.items():
                    result_by_par[pair] = sum(patterns.values())
                order_result = sorted(result_by_par.items(), key=lambda x: x[1], reverse=True)

    return cloze, order_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, help="path to bert model")
    parser.add_argument("-e", "--eval_path", type=str, help="path to datasets")
    parser.add_argument("-o", "--output_path", type=str, help="path to dir output")

    args = parser.parse_args()
    cl, res = main()
