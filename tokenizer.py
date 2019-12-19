import spacy
from spacy.lang.pt import Portuguese

nlp = Portuguese()
# nlp.add_pipe(nlp.create_pipe("tokenizer"))

with open("train.txt") as f:
    for l in f:
        l = l.strip()
        if len(l) == 0:
            continue
        print(" ".join([str(w) for w in nlp(l)]))