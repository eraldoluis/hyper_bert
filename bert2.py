import sys
import datetime
from transformers import BertConfig, BertForMaskedLM, BertTokenizer
import torch
import logging
import argparse
import json
import os
import itertools

logger = logging.getLogger(__name__)


class ClozeBert:
    def __init__(self, model_name):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.config = BertConfig.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=model_name.endswith("-uncased"))
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

    def bert_sentence_score(self, patterns, dataset):
        words_probs_s = {}
        for row in dataset:
            pair = row[0:2]
            words_probs_s[" ".join(row)] = {}
            for pattern in patterns:
                sentences, hyponym_idx, hypernym_idx, idx_mask = self.build_sentences_n_subtoken(pattern, pair)
                idx_all = hyponym_idx + hypernym_idx
                logger.info("Predicting...")
                self.model.eval()
                with torch.no_grad():
                    examples = torch.tensor(sentences, device=self.device)
                    # segments_tensors = torch.tensor([segments_ids])
                    outputs = self.model(examples)  # , segments_tensors)
                predict = outputs[0]
                predict = predict[torch.arange(len(sentences), device=self.device), idx_mask, idx_all]

                predict_hypon = predict[:len(hyponym_idx)]
                predict_hyper = predict[-len(hypernym_idx):]

                words_probs_s[" ".join(row)][pattern] = []
                words_probs_s[" ".join(row)][pattern].append(predict_hypon.cpu().numpy().tolist())
                words_probs_s[" ".join(row)][pattern].append(predict_hyper.cpu().numpy().tolist())

        return words_probs_s

    def bert_sentence_score_multi_pattern(self, patterns, dataset):
        perm_pattern = list(map(list, itertools.permutations(patterns, r=2)))
        words_probs_s = {}
        for row in dataset:
            pair = row[0:2]
            words_probs_s[" ".join(row)] = {}
            for pattern in perm_pattern:
                sentences, hyponym_idx, hypernym_idx, idx_mask, segments_ids = self.build_sentences_n_subtoken_multi_pattern(
                    pattern,
                    pair)
                idx_all = hyponym_idx + hypernym_idx
                idx_all = idx_all * 2
                logger.info("Predicting...")
                self.model.eval()
                with torch.no_grad():
                    examples = torch.tensor(sentences, device=self.device)
                    segments_tensors = torch.tensor(segments_ids, device=self.device)
                    # segments_tensors = torch.tensor([segments_ids])
                    outputs = self.model(examples, token_type_ids=segments_tensors)  # , segments_tensors)
                predict = outputs[0]
                predict = predict[torch.arange(len(sentences), device=self.device), idx_mask, idx_all]

                size = 0
                hipo_1stsen = predict[size:len(hyponym_idx)]
                size += len(hipo_1stsen)
                hiper_1stsen = predict[size:size + len(hypernym_idx)]
                size += len(hiper_1stsen)
                hipo_2ndsen = predict[size:size + len(hyponym_idx)]
                size += len(hipo_2ndsen)
                hiper_2ndsen = predict[size:size + len(hypernym_idx)]

                # [[hipo 1st sen], [hipo 2nd sen], [hyper 1st sen], [hyper 2nd sen]]
                words_probs_s[" ".join(row)]["_".join(pattern)] = []
                words_probs_s[" ".join(row)]["_".join(pattern)].append(hipo_1stsen.cpu().numpy().tolist())
                words_probs_s[" ".join(row)]["_".join(pattern)].append(hipo_2ndsen.cpu().numpy().tolist())
                words_probs_s[" ".join(row)]["_".join(pattern)].append(hiper_1stsen.cpu().numpy().tolist())
                words_probs_s[" ".join(row)]["_".join(pattern)].append(hiper_2ndsen.cpu().numpy().tolist())

        return words_probs_s

    def build_sentences_n_subtoken_multi_pattern(self, patterns, pair):
        logger.info("Tokenizing...")
        hyponym_tokenize = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(pair[0]))
        hypernym_tokenize = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(pair[1]))

        pattern_tokenize1 = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(patterns[0].format("", "").strip()))
        pattern_tokenize2 = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(patterns[1].format("", "").strip()))

        sentences = []
        # Mask 1st sentence
        # mask hyponym
        for i, token_in in enumerate(hyponym_tokenize):
            temp = hyponym_tokenize.copy()
            temp[i] = self.tokenizer.mask_token_id
            sentences.append([self.tokenizer.cls_token_id] + temp + pattern_tokenize1 + hypernym_tokenize +
                             [self.tokenizer.sep_token_id] + hyponym_tokenize + pattern_tokenize2 + hypernym_tokenize
                             + [self.tokenizer.sep_token_id])

        # mask hypernym
        for i, token_in in enumerate(hypernym_tokenize):
            temp = hypernym_tokenize.copy()
            temp[i] = self.tokenizer.mask_token_id
            sentences.append([self.tokenizer.cls_token_id] + hyponym_tokenize + pattern_tokenize1 + temp +
                             [self.tokenizer.sep_token_id] + hyponym_tokenize + pattern_tokenize2 + hypernym_tokenize +
                             [self.tokenizer.sep_token_id])

        # Mask 2nd sentence
        # mask hyponym
        for i, token_in in enumerate(hyponym_tokenize):
            temp = hyponym_tokenize.copy()
            temp[i] = self.tokenizer.mask_token_id
            sentences.append([self.tokenizer.cls_token_id] + hyponym_tokenize + pattern_tokenize1 + hypernym_tokenize +
                             [self.tokenizer.sep_token_id] + temp + pattern_tokenize2 + hypernym_tokenize +
                             [self.tokenizer.sep_token_id])

        # mask hypernym
        for i, token_in in enumerate(hypernym_tokenize):
            temp = hypernym_tokenize.copy()
            temp[i] = self.tokenizer.mask_token_id
            sentences.append([self.tokenizer.cls_token_id] + hyponym_tokenize + pattern_tokenize1 + hypernym_tokenize +
                             [self.tokenizer.sep_token_id] + hyponym_tokenize + pattern_tokenize2 + temp +
                             [self.tokenizer.sep_token_id])

        # get mask_idx
        idx = []
        for sentence in sentences:
            idx.append(sentence.index(self.tokenizer.mask_token_id))

        # segment ids
        seg0 = [0] * (sentences[0].index(self.tokenizer.sep_token_id) + 1)
        seg1 = [1] * (len(sentences[0]) - len(seg0))
        seg = [seg0 + seg1] * len(sentences)

        return sentences, hyponym_tokenize, hypernym_tokenize, idx, seg

    def build_sentences_n_subtoken(self, pattern, pair):
        logger.info("Tokenizing...")
        hyponym_tokenize = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(pair[0]))
        hypernym_tokenize = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(pair[1]))
        pattern_tokenize = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(pattern.format("", "").strip()))

        sentences = []

        # mask hyponym
        for i, token_in in enumerate(hyponym_tokenize):
            temp = hyponym_tokenize.copy()
            temp[i] = self.tokenizer.mask_token_id
            sentences.append([self.tokenizer.cls_token_id] + temp + pattern_tokenize + hypernym_tokenize +
                             [self.tokenizer.sep_token_id])

        # mask hypernym
        for i, token_in in enumerate(hypernym_tokenize):
            temp = hypernym_tokenize.copy()
            temp[i] = self.tokenizer.mask_token_id
            sentences.append([self.tokenizer.cls_token_id] + hyponym_tokenize + pattern_tokenize + temp +
                             [self.tokenizer.sep_token_id])

        # get mask_idx
        idx = []
        for sentence in sentences:
            idx.append(sentence.index(self.tokenizer.mask_token_id))

        return sentences, hyponym_tokenize, hypernym_tokenize, idx


def load_eval_file(f_in):
    eval_data = []
    for line in f_in:
        child_pos, parent_pos, is_hyper, rel = line.strip().split('\t')
        child = child_pos.strip()
        parent = parent_pos.strip()
        is_hyper = is_hyper.strip()
        rel = rel.strip()
        eval_data.append([child, parent, is_hyper, rel])
    # random.shuffle(eval_data)
    return eval_data


def save_bert_file(dict_values, output, dataset_name, model_name, hyper_num, oov_num, f_info_out, save_json,
                   include_oov=True):
    logger.info("save info...")
    f_info_out.write(f'{model_name}\t{dataset_name}\t{len(dict_values)}\t{oov_num}\t{hyper_num}\t{include_oov}\n')
    logger.info("save json...")
    dname = os.path.splitext(dataset_name)[0]
    fjson = json.dumps(dict_values, ensure_ascii=False)
    f = open(os.path.join(output, save_json, dname + ".json"), mode="w", encoding="utf-8")
    f.write(fjson)
    f.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, help="path to bert models", required=True)
    parser.add_argument("-e", "--eval_path", type=str, help="path to datasets", required=True)
    parser.add_argument("-o", "--output_path", type=str, help="path to dir output", required=False)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-l", "--logsoftmax", action="store_true")
    group.add_argument("-z", "--zscore", action="store_true")
    group.add_argument("-x", "--zscore_exp", action="store_true")
    group.add_argument("-b", "--bert_score", action="store_true")

    args = parser.parse_args()
    print("Iniciando bert...")
    cloze_model = ClozeBert(args.model_name)
    try:
        if args.bert_score:
            dir_name = "bert_score"
        elif args.logsoftmax:
            dir_name = "logsoftmax"
        elif args.zscore:
            dir_name = "zscore"
        elif args.zscore_exp:
            dir_name = "zscore_exp"
        else:
            dir_name = "none"
        readable = datetime.datetime.fromtimestamp(int(datetime.datetime.now().timestamp()))
        readable = str(readable).replace(" ", "_")
        dir_name = f'{args.model_name.replace("/", "-")}_{dir_name}_{readable}'
        os.mkdir(os.path.join(args.output_path, dir_name))
    except FileNotFoundError:
        print("Erro na criação do diretório!")
        raise IOError

    f_out = open(os.path.join(args.output_path, dir_name, "info.tsv"), mode="a")
    f_out.write("model\tdataset\tN\toov\thyper_num\tinclude_oov\n")

    patterns = ["{} é um tipo de {}", "{} é um {}", "{} e outros {}", "{} ou outro {}", "{} , um {}"]
    en_patterns = ["{} is a type of {}", "{} is a {}", "{} and others {}", "{} or others {}", "{} , a {}"]

    # 2018 RoolerEtal - Hearst Patterns Revisited
    patterns2 = ["{} que é um exemplo de {}", "{} que é uma classe de {}", "{} que é um tipo de {}",
                 "{} e qualquer outro {}", "{} e algum outro {}", "{} ou qualquer outro {}", "{} ou algum outro {}",
                 "{} que é chamado de {}",
                 "{} é um caso especial de {}",
                 "{} incluindo {}"]

    en_pattern2 = ["{} which is a example of {}", "{} which is a class of {}", "{} which is kind of {}",
                   "{} and any other {}", "{} and some other {}", "{} or any other {}", "{} or some other {}",
                   "{} which is called {}",
                   "{} a special case of {}",
                   "{} including {}"]

    patterns.extend(patterns2)
    en_patterns.extend(en_pattern2)

    en_best_patterns = ['{} or some other {}', '{} or any other {}', '{} and any other {}', '{} is a type of {}', '{} and some other {}', '{} which is kind of {}', '{} a special case of {}', '{} is a {}', '{} which is a example of {}', '{} and others {}', '{} which is called {}', '{} which is a class of {}', '{} or others {}', '{} , a {}', '{} including {}']


    pairs = [['tigre', 'animal', 'True', 'hyper'], ['casa', 'moradia', 'True', 'hyper'],
             ['banana', 'abacate', 'False', 'random']]

    pairs_token_1 = [["banana", "fruta", "True", "hyper"],
                     ['acampamento', 'lugar', 'True', 'hyper'],
                     ['acidente', 'acontecimento', 'True', 'hyper'],
                     ['pessoa', 'discurso', 'False', 'random']]

    # Testes
    # print(f"dataset=TESTE size={len(pairs_token_1)}")
    # if args.bert_score:
    #     logger.info(f"Run BERT score = {args.bert_score}")
    #     # com bert score
    #     result = cloze_model.bert_sentence_score_multi_pattern(patterns[:3], pairs_token_1)
    # else:
    #     result = {}
    # save_bert_file(result, args.output_path, "TESTE", args.model_name, 0, 0, f_out, dir_name, True)
    # logger.info(f"result_size={len(result)}")
    # print(args)
    # f_out.close()

    for file_dataset in os.listdir(args.eval_path):
        if os.path.isfile(os.path.join(args.eval_path, file_dataset)):
            with open(os.path.join(args.eval_path, file_dataset)) as f_in:
                logger.info("Loading dataset ...")
                eval_data = load_eval_file(f_in)
                if args.bert_score:
                    logger.info(f"Run BERT score = {args.bert_score}")
                    result = cloze_model.bert_sentence_score_multi_pattern(en_best_patterns[:4], eval_data)
                    hyper_total = 0
                    oov_num = 0
                save_bert_file(result, args.output_path, file_dataset, args.model_name, hyper_total,
                               oov_num, f_out, dir_name, True)
                logger.info(f"result_size={len(result)}")
    f_out.close()
    logger.info("Done")
    print("Done!")


if __name__ == "__main__":
    main()
