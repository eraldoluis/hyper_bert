import os
import json

print("Iniciando")
path_dataset = "/home/gabrielescobar/dive-pytorch/datasets/validation/ontoPT-validation.tsv"
path_json = "/home/gabrielescobar/Documentos/hyper_bert/teste/neuralmind-bert-base-portuguese-cased_bert-score_n-subtoken/ontoPT-test.json"
dataset = []
print("Carregando JSON")
data_json = json.load(open(path_json))

path_out = "./" + os.path.basename(path_dataset)[:-4] + ".json"

out_data = {}
with open(path_dataset, mode="r", encoding="utf-8") as f:
    for line in f:
        dataset.append(" ".join(line.strip().split("\t")))

for k, v in data_json.items():
    if k in dataset:
        out_data[k] = v

print("Salvando JSON")
with open(path_out, mode='w', encoding="utf-8") as out_f:
    fjson = json.dumps(out_data, ensure_ascii=False)
    out_f.write(fjson)

print("Done!")
