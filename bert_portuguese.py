from transformers import BertConfig, BertForMaskedLM, BertTokenizer, BertModel
import torch
import torch.nn.functional as f
import logging
import numpy as np
import argparse
import random
import json
import os
import itertools

logger = logging.getLogger(__name__)


class ClozeBert:
    def __init__(self, model_name, exp=False, oov=True):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
        self.include_oov = oov
        self.exp = exp

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
        self.z_score = []
        for i in range(20):
            self.z_score.append([0] * 20)

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


    def bert_sentence_score(self, patterns, dataset, vocab_dive, vocab_tokens):
        words_probs_s = {}
        hyper = True
        oov = 0
        hyper_num = 0
        for row in dataset:
            pair = row[0:2]
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
                predict = predict[torch.arange(len(sentences), device=self.device), idx_mask, idx_all]

                predict_hypon = predict[:len(idx_h[0])]
                predict_hyper = predict[- len(idx_h[1]):]

                words_probs_s[" ".join(row)][pattern] = []
                words_probs_s[" ".join(row)][pattern].append(predict_hypon.cpu().numpy().tolist())
                words_probs_s[" ".join(row)][pattern].append(predict_hyper.cpu().numpy().tolist())

        return words_probs_s, hyper_num, oov


    def sentence_score(self, patterns, dataset, vocab_dive, vocab_tokens):
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

            # verificar o tamanho dos subtokens de cada palavra
            size_subtoken_hypo,  size_subtoken_hyper= self.get_len_subtoken(pair)
            for pattern in patterns:
                if not isinstance(self.z_score[size_subtoken_hypo][size_subtoken_hyper], dict):
                    self.z_score[size_subtoken_hypo][size_subtoken_hyper] = {}
                if not pattern in self.z_score[size_subtoken_hypo][size_subtoken_hyper]:
                    self.z_score[size_subtoken_hypo][size_subtoken_hyper][pattern] = self.z_score_1(pattern, tokens_dataset, size_subtoken_hypo, size_subtoken_hyper)

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
                    self.model()
                    outputs = self.model(examples)  # , segments_tensors)
                predict = outputs[0]
                # predict = f.log_softmax(predict, dim=2)

                predict = predict[torch.arange(len(sentences), device=self.device), idx_mask, idx_all]

                if self.exp:
                    # exp no predict
                    predict = torch.exp(predict)

                predict_hypon = predict[:len(idx_h[0])]
                # print(predict_hypon)
                predict_hyper = predict[- len(idx_h[1]):]
                # print(predict_hyper)
                # predict for sentences. shape( len(sentences) )
                # print(predict)

                words_probs_s[" ".join(row)]["z_score"] = self.z_score[size_subtoken_hypo][size_subtoken_hyper].copy()

                words_probs_s[" ".join(row)][pattern] = []
                words_probs_s[" ".join(row)][pattern].append(predict_hypon.cpu().numpy().tolist())
                words_probs_s[" ".join(row)][pattern].append(predict_hyper.cpu().numpy().tolist())
                # words_probs_s[" ".join(row)][pattern] = torch.sum(predict).item()

        return words_probs_s, hyper_num, oov


    def z_score_1(self, pattern, tokens_dataset, len_hypo, len_hyper):
        # calcular para diversos tamanhos de subtoken
        p = pattern.format("", "").strip()
        logger.info("Tokenizing for Z...")
        p_tokenize = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(p))

        sentences_mask_all, idx_mask_all = self.get_sentence_z_score(tokens_dataset,len_hypo, len_hyper, p_tokenize)
        logger.info("Z Score calc...")
        self.model.eval()
        with torch.no_grad():
            examples = torch.tensor(sentences_mask_all, device=self.device)
            # segments_tensors = torch.tensor([segments_ids])
            outputs = self.model(examples)  # , segments_tensors)
        # shape predict (2*dataset_words, sentence_len, vocab_bert)
        predict = outputs[0]
        # predict = f.log_softmax(predict, dim=2)

        tensor_tokens_dataset = torch.tensor(tokens_dataset, device=self.device).unsqueeze(dim=1)
        idx_mask_tensor = torch.tensor(idx_mask_all, device=self.device)
        predict = predict[torch.arange(len(sentences_mask_all), device=self.device), idx_mask_tensor, tensor_tokens_dataset]
        values, indices = torch.topk(torch.topk(predict, k=1).values.view(-1), k=5)
        logger.info(f"MAX values for z {values}")
        if self.exp:
            # exp no zscore
            predict = torch.exp(predict)
        soma = predict.sum().item()
        return soma


    def get_len_subtoken(self, pair):
        hyponym = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(pair[0]))
        hypernym = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(pair[1]))
        return len(hyponym), len(hypernym)


    def get_sentence_z_score(self, tokens_dataset, len_hypo, len_hyper, pattern):
        size = len_hypo + len_hyper
        comb_obj = itertools.product(tokens_dataset, repeat=size-1)
        comb_list = list(map(list, comb_obj))
        sentences = []

        for i in range(size):
            for j in range(len(comb_list)):
                sentence = comb_list[j].copy()
                sentence.insert(i, self.tokenizer.mask_token_id)
                sentences.append(sentence)
        sentences = np.array(sentences)

        init_sentence = [[self.tokenizer.cls_token_id]] * len(sentences)
        end_sentence = [[self.tokenizer.sep_token_id]] * len(sentences)
        pattern_sentence = [pattern] * len(sentences)

        sentence_hypo = sentences[:, :len_hypo]
        sentence_hyper = sentences[:, len_hypo:]

        sentences_prod = np.concatenate((init_sentence, sentence_hypo, pattern_sentence, sentence_hyper, end_sentence), axis=1)
        idx_mask= np.where(sentences_prod == self.tokenizer.mask_token_id)
        idx_mask = idx_mask[1].tolist()
        return sentences_prod.tolist(), idx_mask


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
            vocab.extend(self.tokenizer.tokenize(data[0]))
            vocab.extend(self.tokenizer.tokenize(data[1]))
        # removendo tokens repetidos
        vocab = set(vocab)
        vocab = list(vocab)
        logger.info("Tokenizando vocab...")
        vocab_tokenize = self.tokenizer.convert_tokens_to_ids(vocab)
        return vocab_tokenize


    def subtoken_dataset(self, dataset):
        size = []
        for i in range(20):
            size.append([0] * 20)

        for pair in dataset:
            hypo, hyper = self.get_len_subtoken(pair[:2])
            size[hypo][hyper] =  size[hypo][hyper] + 1

        return size

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


def save_bert_file(dict, output, dataset_name, model_name, hyper_num, oov_num, f_info_out, include_oov=True):
    logger.info("save info...")
    f_info_out.write(f'{model_name}\t{dataset_name}\t{len(dict)}\t{oov_num}\t{hyper_num}\t{include_oov}\n')
    logger.info("save json...")
    dname = os.path.splitext(dataset_name)[0]
    fjson = json.dumps(dict, ensure_ascii=False)
    f = open(os.path.join(output, model_name.replace("/", "-"), dname + ".json"), mode="w", encoding="utf-8")
    f.write(fjson)
    f.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, help="path to bert models", required=True)
    parser.add_argument("-e", "--eval_path", type=str, help="path to datasets", required=True)
    parser.add_argument("-o", "--output_path", type=str, help="path to dir output", required=False)
    parser.add_argument("-v", "--vocab", type=str, help="dir of vocab", required=False)
    parser.add_argument("-u", "--include_oov", action="store_true", help="to include oov on results",
                        default=True)  # sempre True

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-l", "--logsoftmax", action="store_true")
    group.add_argument("-z", "--zscore", action="store_true")
    group.add_argument("-x", "--zscore_exp", action="store_true")
    group.add_argument("-b", "--bert_score", action="store_true")

    args = parser.parse_args()
    print("Iniciando bert...")
    cloze_model = ClozeBert(args.model_name, args.zscore_exp)
    try:
        os.mkdir(os.path.join(args.output_path, args.model_name.replace("/", "-")))
    except:
        pass

    f_out = open(os.path.join(args.output_path, args.model_name.replace('/', '-'), "info.tsv"), mode="a")
    f_out.write("model\tdataset\tN\toov\thyper_num\tinclude_oov\n")

    patterns = ["{} é um tipo de {}", "{} é um {}", "{} e outros {}", "{} ou outro {}", "{} , um {}"]

    # 2018 RoolerEtal - Hearst Patterns Revisited
    patterns2 = ["{} que é um exemplo de {}", "{} que é uma classe de {}", "{} que é um tipo de {}",
                 "{} e qualquer outro {}", "{} e algum outro {}", "{} ou qualquer outro {}", "{} ou algum outro {}",
                 "{} que é chamado de {}",
                 "{} é um caso especial de {}",
                 "{} incluindo {}"]
    # patterns = ["[MASK] é um tipo de [MASK]", "[MASK] é um [MASK]"]

    patterns.extend(patterns2)

    pairs = [['tigre', 'animal', 'True', 'hyper'], ['casa', 'moradia', 'True', 'hyper'],
             ['banana', 'abacate', 'False', 'random']]

    pairs_token_1 = [['acampamento', 'lugar', 'True', 'hyper'],
                     ['acidente', 'acontecimento', 'True', 'hyper'],
                     ['pessoa', 'discurso', 'False', 'random']]
                     # ['pessoa', 'discurso', 'False', 'random'],
                     # ["banana", "fruta", "True", "hyper"]]

    # logger.info("Loading vocab dive ...")
    # dive_vocab = []
    # with open(os.path.join(args.vocab, "vocab.txt"), mode="r", encoding="utf-8") as f_vocab:
    #     for line in f_vocab:
    #         word, count = line.strip().split()
    #         dive_vocab.append(word)

    # # Testes
    # print(f"dataset=TESTE size={len(pairs_token_1)}")
    # vocab_dataset_tokens = cloze_model.get_tokens_dataset(pairs_token_1)
    # if args.zscore or args.zscore_exp:
    #     logger.info(f"Run Z Score = {args.zscore}")
    #     logger.info(f"Run Z Score_EXP = {args.zscore_exp}")
    #     # com zscore
    #     result, hyper_total, oov_num = cloze_model.z_sentence_score(patterns, pairs_token_1, [], vocab_dataset_tokens)
    # #
    # if args.logsoftmax:
    #     logger.info(f"Run Log Softmax = {args.logsoftmax}")
    #     # bert score
    #     result, hyper_total, oov_num = cloze_model.sentence_score(patterns, pairs_token_1, [], vocab_dataset_tokens)
    #
    # if args.bert_score:
    #     logger.info(f"Run BERT score = {args.bert_score}")
    #     # com bert score
    #     result, hyper_total, oov_num = cloze_model.bert_sentence_score(patterns, pairs_token_1, [], vocab_dataset_tokens)
    # save_bert_file(result, args.output_path, "TESTE", args.model_name.replace('/', '-'), hyper_total, oov_num,
    #                f_out, args.include_oov)
    # logger.info(f"result_size={len(result)}")
    print(args)

    for file_dataset in os.listdir(args.eval_path):
        if os.path.isfile(os.path.join(args.eval_path, file_dataset)):
            with open(os.path.join(args.eval_path, file_dataset)) as f_in:
                logger.info("Loading dataset ...")
                eval_data = load_eval_file(f_in)
                vocab_dataset_tokens = cloze_model.get_tokens_dataset(eval_data)
                # com bert score
                if args.bert_score:
                    logger.info(f"Run BERT score = {args.bert_score}")
                    result, hyper_total, oov_num = cloze_model.bert_sentence_score(patterns, eval_data, [], vocab_dataset_tokens)
                # com zscore
                if args.zscore or args.zscore_exp:
                    logger.info(f"Run Z Score = {args.zscore}")
                    result, hyper_total, oov_num = cloze_model.z_sentence_score(patterns, eval_data, [], vocab_dataset_tokens)
                # com log_softmax
                if args.logsoftmax:
                    logger.info(f"Run Log Softmax = {args.logsoftmax}")
                    result, hyper_total, oov_num = cloze_model.sentence_score(patterns, eval_data, [], vocab_dataset_tokens)

                save_bert_file(result, args.output_path, file_dataset, args.model_name.replace('/', '-'), hyper_total,
                               oov_num, f_out, args.include_oov)
                logger.info(f"result_size={len(result)}")
    f_out.close()
    logger.info("Done")
    print("Done!")


def subtoken_size():
    print("Iniciando bert...")
    model_name = "neuralmind/bert-base-portuguese-cased"
    cloze_model = ClozeBert(model_name)
    out_path = "results/subtoken_size/subtoken_dataset.tsv"
    dataset_path = "/home/gabrielescobar/Documentos/dive-pytorch/datasets"
    f_out = open(out_path, encoding="utf-8", mode="w")
    f_out.write("model\tdataset\tsubtoken\tN\n")

    pairs_token_1 = [['acampamento', 'lugar', 'True', 'hyper'],
                     ['acidente', 'acontecimento', 'True', 'hyper'],
                     ['pessoa', 'discurso', 'False', 'random'],
                     ['pessoa', 'discurso', 'False', 'random'],
                     ["banana", "fruta", "True", "hyper"]]


    for file_dataset in os.listdir(dataset_path):
        if os.path.isfile(os.path.join(dataset_path, file_dataset)):
            with open(os.path.join(dataset_path, file_dataset)) as f_in:
                logger.info("Loading dataset ...")
                eval_data = load_eval_file(f_in)
                size = cloze_model.subtoken_dataset(eval_data)
                for i in range(len(size)):
                    for j in range(len(size)):
                        if size[i][j] != 0:
                            f_out.write(f"{model_name}\t{file_dataset}\t{i},{j}\t{size[i][j]}\n")

    f_out.close()
    return cloze_model

if __name__ == "__main__":
    # main()
    m = subtoken_size()

'''
size tokens dataset  = 2723
'''