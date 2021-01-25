from collections import Counter
import argparse
import random
from bert2 import ClozeBert


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
    # selecionar os pares com tamanhos determinados
    word_length_tokenize = {}
    for i in range(2, 15):
        max_number = 10000
        word_length_tokenize[i] = []
        while max_number > 0:
            sample = random.choice(list(freq.items()))
            sample2 = random.choice(list(freq.items()))
            #TODO



    return cloze_model, counter_length


if __name__ == '__main__':
    model, freq = main()
    # EN params
    # -m bert-base-uncased -l ./UKWAC_frequency_words.txt -c 5
