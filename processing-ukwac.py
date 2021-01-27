from collections import Counter
path_local = "/mnt/Data/Downloads-2/ukwac_subset_100M.txt"

# stopword from dive
path_stopword = "./stop_word_list"
stopword_list = set(open(path_stopword).read().splitlines())
count_lines = 0
word_count = 0
word_count_clean = 0
counter = Counter()
with open(path_local, mode="r", encoding="ISO-8859-1") as f:
    for line in f:
        if not line.startswith("CURRENT URL"):
            word_count += len(line.split())
            words_clean = [w for w in line.lower().strip().split() if w not in stopword_list]
            words_clean = [w for w in words_clean if w.isalpha()]
            word_count_clean += len(words_clean)
            counter.update(words_clean)
            count_lines += 1


print(f"Há {count_lines} linhas no arquivo")
print(f"Com {word_count} palavras")
print(f"Dessas, {word_count_clean} palavras foram usadas")

# escrevendo esse dicionario em um csv
most_commom = counter.most_common()
path_out = "./UKWAC_frequency_words.txt"

with open(path_out, mode="w", encoding="utf8") as f_out:
    for (word, freq) in most_commom:
        f_out.write(f"{word}\t{freq}\n")

print(f"Arquivo escrito com {len(most_commom)} palavras")
