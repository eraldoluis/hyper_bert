from collections import Counter
import argparse
import random
from bert2 import ClozeBert


def escrever_random_pares(word_length_tokenize):
    path = "./random_pairs.csv"
    with open(path, mode="w", encoding="utf8") as f:
        for length, data in word_length_tokenize.items():
            for pair1, pair2 in data:
                f.write(f"{pair1[0]}\t{pair1[1]}\t{pair2[0]}\t{pair2[1]}\n")


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

    status = {}
    for i in range(2, 15):
        status[i] = False

    empty_length = True
    word_length_tokenize = {}
    while empty_length:
        sample = random.choice(list(word_length.items()))
        sample2 = random.choice(list(word_length.items()))
        len_sample = sample[1] + sample2[1]
        print(len_sample)
        if len_sample in word_length_tokenize.keys():
            if not status[len_sample]:
                if [sample, sample2] not in word_length_tokenize[len_sample]:
                    word_length_tokenize[len_sample].append([sample, sample2])
                    if len(word_length_tokenize[len_sample]) >= 10000:
                        status[len_sample] = True
                        print(f"Len {len_sample} chegou a 10000!")
        elif len_sample < 15:
            word_length_tokenize[len_sample] = []
            word_length_tokenize[len_sample].append([sample, sample2])

        if all(status.values()):
            break

    escrever_random_pares(word_length_tokenize)

    return cloze_model, word_length_tokenize


if __name__ == '__main__':
    main()
    # EN params
    # -m bert-base-uncased -l ./UKWAC_frequency_words.txt -c 3
