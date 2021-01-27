from collections import Counter
from collections import defaultdict

from itertools import combinations_with_replacement
import argparse
import random
from bert2 import ClozeBert


def escrever_random_pares(word_length_tokenize):
    path = "./random_pairs.csv"
    with open(path, mode="w", encoding="utf8") as f:
        for data in word_length_tokenize:
            f.write(f"{data[0]}\t{data[1]}\t{data[2]}\t{data[3]}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, help="path to bert models", required=True)
    parser.add_argument("-l", "--list_word", type=str, help="path to list_words", required=True)
    parser.add_argument("-c", "--min_frequency", type=int, help="frequency word >= min_frequency", required=True)
    args = parser.parse_args()

    print("Iniciando bert...")
    cloze_model = ClozeBert(args.model_name)
    count_threshold = args.min_frequency
    words = {}
    with open(args.list_word, mode="r", encoding="utf8") as f:
        for line in f:
            word, cnt = line.strip().split("\t")
            cnt = int(cnt)
            if word in words:
                raise KeyError
            elif cnt >= count_threshold:
                words[word] = cnt
    print(f"Há {len(words)} palavras")
    print(f"Com frequência maior que {count_threshold}")

    # pegar o comprimento de cada palavra conforme o wordpiece
    word_length = {}
    for word, count in words.items():
        if word in word_length:
            raise KeyError
        else:
            word_length[word] = len(cloze_model.tokenizer.tokenize(word))
    counter_length = Counter(word_length)
    inv_word_len = defaultdict(list)
    _ = {inv_word_len[v].append(k) for k, v in word_length.items()}

    # Reduzir o campo de busca
    sizes = set(word_length.values())
    candidatos = list(combinations_with_replacement(sizes, 2))
    candidatos_dict = {}
    for size in candidatos:
        len_sample = sum(size)
        if len_sample > 15:
            continue
        if len_sample in candidatos_dict:
            candidatos_dict[len_sample].append(size)
        else:
            candidatos_dict[len_sample] = []
            candidatos_dict[len_sample].append(size)
    del candidatos

    word_len_tokenize = []
    status = {}
    for size, combs in candidatos_dict.items():
        # size, combs = comprimento par, ex: 4: (1,3), (2,2)
        status[size] = 0
        while status[size] < 10:
            comb = random.choice(combs) # (1,3)
            idx = random.choice([0, 1]) # indice da tupla acima
            if idx == 1:
                hipo_len, hiper_len = comb[1], comb[0]
            else:
                hipo_len, hiper_len = comb[0], comb[1]
            hipo_word = random.choice(inv_word_len[hipo_len])
            hiper_word = random.choice(inv_word_len[hiper_len])

            if (hipo_word, hipo_len, hiper_word, hiper_len) not in word_len_tokenize:
                word_len_tokenize.append((hipo_word, hipo_len, hiper_word, hiper_len))
                status[size] += 1
        print(f"Terminado o status {size}")

    # escrever_random_pares(word_len_tokenize)

    return cloze_model, candidatos_dict, word_length, inv_word_len


if __name__ == '__main__':
    m, candi, freq, inv = main()
    # EN params
    # -m bert-base-uncased -l ./UKWAC_frequency_words.txt -c 3
