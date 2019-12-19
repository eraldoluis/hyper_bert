import spacy
from spacy.gold import biluo_tags_from_offsets


nlp = spacy.load("en_core_web_sm")

with open("lucas.txt") as f:
    txt = " ".join([l.strip() for l in f.readlines() if len(l.strip()) > 0])

doc = nlp(txt)



ents = [(e.start_char, e.end_char, e.label_) for e in doc.ents]

tags = biluo_tags_from_offsets(doc, ents)

# print("\n".join([str((t, e)) for t, e in zip(doc, tags)]))
print("\n".join([str((t, t.ent_iob_ if t.ent_iob_ in ("O", "") else t.ent_iob_ + "-" + t.ent_type_)) for t in doc]))