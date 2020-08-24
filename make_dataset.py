import argparse
import os
import logging

from bert_portuguese import ClozeBert
from bert_portuguese import load_eval_file

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def write_dataset(data, name_dataset, path_out):
    logger.info(f"salvando dataset {name_dataset}")
    with open(os.path.join(path_out, name_dataset + "_token_1.tsv"), mode="w", encoding="utf-8") as f_out:
        for row in data:
            f_out.write("\t".join(row) + "\n")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, help="path to bert models", required=True)
    parser.add_argument("-e", "--eval_path", type=str, help="path to datasets", required=True)
    parser.add_argument("-o", "--output_path", type=str, help="path to dir output", required=False)
    args = parser.parse_args()

    logger.info("Iniciando make_dataset...")
    bert = ClozeBert(args.model_name)
    print(bert.tokenizer.tokenize("banana"))
    print(bert.tokenizer.tokenize("tigre"))

    for name in os.listdir(args.eval_path):
        if os.path.isfile(os.path.join(args.eval_path, name)):
            if name == "conceptnet-hypernym-1.tsv":
                new_data = []
                with open(os.path.join(args.eval_path, name), mode="r", encoding="utf-8") as f:
                    data = load_eval_file(f)
                    for row in data:
                        h_0, h_1 = bert.tokenizer.tokenize(row[0]), bert.tokenizer.tokenize(row[1])
                        if len(h_0) == 1 and len(h_1) == 1:
                            new_data.append(row)
                write_dataset(new_data, name[:-4], args.output_path)

    return bert, new_data


if __name__ == '__main__':
    bert, data = main()
