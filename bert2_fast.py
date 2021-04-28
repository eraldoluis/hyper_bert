import datetime
from transformers import BertConfig, BertForMaskedLM, BertTokenizer
import torch
import logging
import argparse
import operator
import json
import os
import numpy as np
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
        self.words_probs_s = {}

    def batch(self, dataset):
        len_dict = {}
        for row in dataset:
            self.words_probs_s["\t".join(row)] = {}
            pair = row[0:2]
            hyponym_tokenize = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(pair[0]))
            hypernym_tokenize = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(pair[1]))
            comprimento = len(hyponym_tokenize) + len(hypernym_tokenize)
            if comprimento in len_dict.keys():
                len_dict[comprimento].append((row, [len(hyponym_tokenize), len(hypernym_tokenize)]))
            else:
                len_dict[comprimento] = []
                len_dict[comprimento].append((row, [len(hyponym_tokenize), len(hypernym_tokenize)]))

        return len_dict

    def build_sentences_mask_all(self, pattern, pair):
        logger.info("Tokenizing MASK ALL...")
        hyponym_tokenize = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(pair[0]))
        hypernym_tokenize = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(pair[1]))
        pattern_tokenize = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(pattern.format("", "").strip()))
        model_sentence = [self.tokenizer.cls_token_id] + hyponym_tokenize + pattern_tokenize \
                         + hypernym_tokenize + [self.tokenizer.sep_token_id]
        sentences = []
        for i in range(1, len(model_sentence) - 1):
            tmp = model_sentence.copy()
            tmp[i] = self.tokenizer.mask_token_id
            sentences.append(tmp)

        idx = np.arange(1, len(model_sentence) - 1).tolist()

        return sentences, model_sentence[1:-1], idx

    def build_sentences_mask_all_batch(self, pattern, pairs, comprimento):
        # todos os pares têm o mesmo comprimento
        hyponym_array = []
        hypernym_array = []
        batch_size = len(pairs)

        pattern_tokenize = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(pattern.format("", "").strip()))
        pattern_tensor = torch.tensor([pattern_tokenize], device=self.device)
        sentence_all = torch.empty(0, comprimento + len(pattern_tokenize), dtype=torch.int64)
        token_find_all = torch.empty(0, comprimento + len(pattern_tokenize), dtype=torch.int64)

        for pair in pairs:
            hyponym_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(pair[0]))
            hypernym_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(pair[1]))

            hyponym_array.append(hyponym_token)
            hypernym_array.append(hypernym_token)

            hyponym_tensor = torch.tensor([hyponym_token], device=self.device)
            hypernym_tensor = torch.tensor([hypernym_token], device=self.device)
            sentence = torch.cat((hyponym_tensor, pattern_tensor, hypernym_tensor), 1)
            sentence_len = sentence.size()[-1]

            sentence = sentence.repeat(sentence_len, 1)
            token_find = sentence.diag()
            sentence.fill_diagonal_(self.tokenizer.mask_token_id)

            token_find_all = torch.cat((token_find_all, token_find.unsqueeze(dim=0)), 0)
            sentence_all = torch.cat((sentence_all, sentence), 0)

        idx_token = torch.arange(1, comprimento + len(pattern_tokenize) + 1)
        idx_token = idx_token.repeat(len(sentence_all), 1)

        cls = torch.tensor([[self.tokenizer.cls_token_id]])
        sep = torch.tensor([[self.tokenizer.sep_token_id]])
        cls = torch.cat([cls] * len(sentence_all))
        sep = torch.cat([sep] * len(sentence_all))

        sentences_predict = torch.cat((cls, sentence_all, sep), 1)
        return sentences_predict, token_find_all, idx_token

    def bert_maskall(self, patterns, dataset, batch_size=2):
        self.words_probs_s = {}
        batch_dataset = self.batch(dataset)
        for comprimento, paresComprimento in batch_dataset.items():
            for mini_batch in range(0, len(paresComprimento), batch_size):
                logger.info(f"Predicting MASK_ALL... {mini_batch}")
                batch_array = paresComprimento[mini_batch: mini_batch + batch_size]
                pairs_batch_array = [p[0] for p in batch_array]
                for pattern in patterns:
                    sentences, token_id, idx_token = self.build_sentences_mask_all_batch(pattern, pairs_batch_array,
                                                                                         comprimento)
                    self.model.eval()
                    with torch.no_grad():
                        segments_tensors = torch.zeros(len(sentences), len(sentences[0]), device=self.device)
                        outputs = self.model(sentences, segments_tensors)  # , segments_tensors)
                    predict = outputs[0]
                    idx_token = idx_token[0:len(pairs_batch_array)].view(1, -1).squeeze(dim=0)
                    token_id = token_id.view(1, -1).squeeze(dim=0)
                    predict2 = predict[torch.arange(len(sentences), device=self.device), idx_token, token_id]
                    sentences_tensor = predict2.split(int(len(sentences) / len(pairs_batch_array)), dim=0)
                    for s in range(len(sentences_tensor)):
                        self.words_probs_s["\t".join(batch_array[s][0])]["comprimento"] = batch_array[s][1]
                        if pattern in self.words_probs_s["\t".join(batch_array[s][0])]:
                            self.words_probs_s["\t".join(batch_array[s][0])][pattern] = sentences_tensor[
                                s].cpu().numpy().tolist()
                        else:
                            self.words_probs_s["\t".join(batch_array[s][0])][pattern] = sentences_tensor[
                                s].cpu().numpy().tolist()

        return self.words_probs_s


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


def save_bert_file(dict_values, output, dataset_name, save_json):
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
    parser.add_argument("-b", "--batch_size", type=str, help="batch pair number", required=True)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--bert_score_maskall", action="store_true")

    group_language = parser.add_mutually_exclusive_group()
    group_language.add_argument("--pt", action="store_true")
    group_language.add_argument("--en", action="store_true")
    args = parser.parse_args()
    print("Iniciando bert...")
    cloze_model = ClozeBert(args.model_name)
    try:
        if args.bert_score_maskall:
            dir_name = "bert_score_maskall"
        else:
            dir_name = "none"
        readable = datetime.datetime.fromtimestamp(int(datetime.datetime.now().timestamp()))
        readable = str(readable).replace(" ", "_")
        dir_name = f'{args.model_name.replace("/", "-")}_{dir_name}_{readable}'
        os.mkdir(os.path.join(args.output_path, dir_name))
    except FileNotFoundError:
        print("Erro na criação do diretório!")
        raise FileNotFoundError

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

    pairs = [['tigre', 'animal', 'True', 'hyper'], ['casa', 'moradia', 'True', 'hyper'],
             ['banana', 'abacate', 'False', 'random']]

    pairs_token_1 = [["banana maça", "fruta", "True", "hyper"],
                     ['acampamento', 'lugar', 'True', 'hyper'],
                     ['acidente', 'acontecimento', 'True', 'hyper'],
                     ['pessoa', 'discurso', 'False', 'random']]

    if args.pt:
        pattern_train = patterns
    elif args.en:
        pattern_train = en_patterns
    else:
        raise ValueError

    # Testes
    # print(f"dataset=TESTE size={len(pairs_token_1)}")
    # if args.bert_score_maskall:
    #     logger.info(f"Run BERT MASK ALL= {args.bert_score_maskall}")
    #     result = cloze_model.bert_maskall(pattern_train[:3], pairs, int(args.batch_size))
    # else:
    #     logger.info(f"nenhum método selecionado")
    #     raise ValueError
    # save_bert_file(result, args.output_path, "TESTE", dir_name)
    # logger.info(f"result_size={len(result)}")
    # print(args)

    for file_dataset in os.listdir(args.eval_path):
        if os.path.isfile(os.path.join(args.eval_path, file_dataset)):
            with open(os.path.join(args.eval_path, file_dataset)) as f_in:
                logger.info("Loading dataset ...")
                eval_data = load_eval_file(f_in)
                if args.bert_score_maskall:
                    logger.info(f"Run BERT MASK ALL= {args.bert_score_maskall}")
                    hyper_total = 0
                    oov_num = 0
                    result = cloze_model.bert_maskall(pattern_train, eval_data, int(args.batch_size))
                else:
                    logger.info(f"Nenhum método selecionado")
                    raise ValueError
                save_bert_file(result, args.output_path, file_dataset, dir_name)
                logger.info(f"result_size={len(result)}")
    logger.info("Done")
    print("Done!")


if __name__ == "__main__":
    main()
