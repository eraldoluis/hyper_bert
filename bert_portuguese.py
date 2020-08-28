from transformers import BertConfig, BertForMaskedLM, BertTokenizer, BertModel
import torch
import torch.nn.functional as f
import logging
import numpy as np
import argparse
import random
import json
import os

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
                words_probs_s[" ".join(row)][pattern].append(predict_hypon.cpu().numpy().tolist())
                words_probs_s[" ".join(row)][pattern].append(predict_hyper.cpu().numpy().tolist())
                # words_probs_s[" ".join(row)][pattern] = torch.sum(predict).item()

        return words_probs_s, hyper_num, oov


    def z_sentence_score(self, patterns, dataset, vocab_dive, tokens_dataset):
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
                z_score = self.z_build_sentences(pattern, tokens_dataset)
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
                # predict = f.log_softmax(predict, dim=2)
                predict = predict[torch.arange(len(sentences), device=self.device), idx_mask, idx_all]

                predict_hypon = predict[:len(idx_h[0])]
                # print(predict_hypon)
                predict_hyper = predict[- len(idx_h[1]):]
                # print(predict_hyper)
                # predict for sentences. shape( len(sentences) )
                # print(predict)
                if "z_score" not in words_probs_s[" ".join(row)]:
                    words_probs_s[" ".join(row)]["z_score"] = {}
                    words_probs_s[" ".join(row)]["z_score"][pattern] = z_score.item()
                else:
                    words_probs_s[" ".join(row)]["z_score"][pattern] = z_score.item()
                words_probs_s[" ".join(row)][pattern] = []
                words_probs_s[" ".join(row)][pattern].append(predict_hypon.cpu().numpy().tolist())
                words_probs_s[" ".join(row)][pattern].append(predict_hyper.cpu().numpy().tolist())
                # words_probs_s[" ".join(row)][pattern] = torch.sum(predict).item()

        return words_probs_s, hyper_num, oov

    def z_build_sentences(self, pattern, tokens_dataset):  # feito, agora falta tratar onde isso eh chamado
        p = pattern.format("", "").strip()
        logger.info("Tokenizing...")
        p_tokenize = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(p))

        # pegar indice das masks
        idx_masks = []
        senteces_mask_all = []
        # fixar inicio e mascarar final
        for t_v in tokens_dataset:
            sentence = [self.tokenizer.cls_token_id] + [t_v] + p_tokenize + [self.tokenizer.mask_token_id] + [self.tokenizer.sep_token_id]
            idx_masks.append(sentence.index(self.tokenizer.mask_token_id))
            senteces_mask_all.append(sentence)

        # fixar final e mascarar inicio
        for t_v in tokens_dataset:
            sentence = [self.tokenizer.cls_token_id] + [self.tokenizer.mask_token_id] + p_tokenize + [t_v] + [self.tokenizer.sep_token_id]
            idx_masks.append(sentence.index(self.tokenizer.mask_token_id))
            senteces_mask_all.append(sentence)

        logger.info("Z Score calc...")
        self.model.eval()
        with torch.no_grad():
            examples = torch.tensor(senteces_mask_all, device=self.device)
            # segments_tensors = torch.tensor([segments_ids])
            outputs = self.model(examples)  # , segments_tensors)
        predict = outputs[0]
        # predict = f.log_softmax(predict, dim=2)
        predict = predict[torch.arange(len(senteces_mask_all), device=self.device), idx_masks, tokens_dataset * 2]
        soma = predict.sum()
        return soma


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

    def get_tokens_dataset(self, pairs_token_1):
        vocab = []
        for data in pairs_token_1:
            vocab.append(data[0])
            vocab.append(data[1])
        logger.info("Tokenizando vocab...")
        vocab_tokenize = self.tokenizer.convert_tokens_to_ids(vocab)
        return vocab_tokenize


def load_eval_file(f_in):
    eval_data = []
    for line in f_in:
        child_pos, parent_pos, is_hyper, rel = line.strip().split('\t')
        child = child_pos.strip()
        parent = parent_pos.strip()
        is_hyper = is_hyper.strip()
        rel = rel.strip()
        eval_data.append([child, parent, is_hyper, rel])
    #random.shuffle(eval_data)
    return eval_data


def save_bert_file(dict, output, dataset_name, model_name, hyper_num, oov_num, f_info_out, include_oov = True):
    logger.info("save info...")
    f_info_out.write(f'{model_name}\t{dataset_name}\t{len(dict)}\t{oov_num}\t{hyper_num}\t{include_oov}\n')
    logger.info("save json...")
    dname = os.path.splitext(dataset_name)[0]
    fjson = json.dumps(dict, ensure_ascii=False)
    f = open(os.path.join(output, model_name.replace("/","-"), dname + ".json"), mode="w", encoding="utf-8")
    f.write(fjson)
    f.close()


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

    pairs_token_1 = [['acampamento', 'lugar', 'True', 'hyper'],
                     ['acidente', 'acontecimento', 'True', 'hyper'],
                     ['pessoa', 'discurso', 'False', 'random']]

    # logger.info("Loading vocab dive ...")
    # dive_vocab = []
    # with open(os.path.join(args.vocab, "vocab.txt"), mode="r", encoding="utf-8") as f_vocab:
    #     for line in f_vocab:
    #         word, count = line.strip().split()
    #         dive_vocab.append(word)

    # print(f"dataset=TESTE size={len(pairs_token_1)}")
    # vocab_dataset_tokens = cloze_model.get_tokens_dataset(pairs_token_1)
    # result, hyper_total, oov_num = cloze_model.z_sentence_score(patterns, pairs_token_1, [], vocab_dataset_tokens)
    # save_bert_file(result, args.output_path, "TESTE", args.model_name.replace('/', '-'), hyper_total, oov_num,
    #                f_out, args.include_oov)
    # logger.info(f"result_size={len(result)}")

    for file_dataset in os.listdir(args.eval_path):
        if os.path.isfile(os.path.join(args.eval_path, file_dataset)):
            with open(os.path.join(args.eval_path, file_dataset)) as f_in:
                logger.info("Loading dataset ...")
                eval_data = load_eval_file(f_in)
                vocab_dataset_tokens = cloze_model.get_tokens_dataset(eval_data)
                result, hyper_total, oov_num = cloze_model.z_sentence_score(patterns, eval_data, [],
                                                                            vocab_dataset_tokens)
                save_bert_file(result, args.output_path, file_dataset, args.model_name.replace('/', '-'), hyper_total,
                               oov_num, f_out, args.include_oov)
                # eval_data = load_eval_file(f_in)
                # print(f"dataset={file_dataset} size={len(eval_data)}")
                # result, hyper_total, oov_num = cloze_model.sentence_score(patterns, eval_data[:10], dive_vocab)
                # save_bert_file(result, args.output_path, file_dataset, args.model_name.replace('/', '-'), hyper_total, oov_num, f_out, args.include_oov)
                logger.info(f"result_size={len(result)}")
    f_out.close()
    logger.info("Done")
    print("Done!")


if __name__ == "__main__":
    main()
