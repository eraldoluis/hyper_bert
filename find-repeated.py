import os
import collections
path_d = '/home/gabriel/Documentos/dive-pytorch/datasets'
data = []
rep = []
r = []
for filedataset in os.listdir(path_d):
    r = []
    data = []
    if os.path.isfile(os.path.join(path_d, filedataset)):
        with open(os.path.join(path_d, filedataset)) as f_in:
            for line in f_in:
                data.append(line.strip())
            if len(data) != len(set(data)):
                print(f"{filedataset} com {len(data)} e {len(set(data))}")

        r.append(item for item, count in collections.Counter(data).items() if count > 1)
        print([item for item, count in collections.Counter(data).items() if count > 1])
        rep.append([item for item, count in collections.Counter(data).items() if count > 1])