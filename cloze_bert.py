# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import logging
import random

import numpy as np
import torch
from transformers import BertConfig, BertForMaskedLM, BertTokenizer

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class ClozeBert:
    def __init__(self, model_name):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
        self.config = BertConfig.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=model_name.endswith("-uncased"))
        self.model = BertForMaskedLM.from_pretrained(model_name, config=self.config)

    def most_probabable_words(self, texts):
        words_probs_s = []
        for text in texts:
            logger.info("Tokenizing...")
            tokenized_text = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
            example = self.tokenizer.build_inputs_with_special_tokens(tokenized_text)

            idx_mask = example.index(self.tokenizer.mask_token_id)

            logger.info("Predicting...")
            self.model.eval()
            with torch.no_grad():
                examples = torch.tensor([example])
                outputs, = self.model(examples)

            # outputs shape is (batch_example, words, scores).
            probs_mask = outputs[0, idx_mask]

            logger.info("Zipping...")
            words_probs = zip(probs_mask, self.tokenizer.vocab.keys())

            logger.info("Sorting...")
            # pair words with their scores (score, word) and sort them by score
            words_probs = sorted(words_probs, reverse=True)

            words_probs_s.append(words_probs)

        return words_probs_s


def main():
    # Set seed
    # set_seed(13)

    cloze = ClozeBert("bert-base-multilingual-cased")

    # texts = ["Carros são usados para se locomover . Carro é um tipo de [MASK] [MASK] .",
    #          "Carros são usados para se locomover . Carro é um tipo de [MASK] .",
    #          "As pessoas vivem em casas . Casa é um tipo de [MASK] .",
    #          "As pessoas vivem em casas . Casa é um tipo de [MASK] [MASK] ."]
    texts = ["Carro é um tipo de [MASK] [MASK] .",
             "Casa é um tipo de [MASK] .",
             "avião é um [MASK] .",
             "notebook é um [MASK] .",
             "gabriel é um [MASK] ."
             ]


    words_probs_s = cloze.most_probabable_words(texts)

    for words_probs, text in zip(words_probs_s, texts):
        print(cloze.tokenizer.tokenize(text))

        # (rank, (score, word))
        words_probs = list(zip(range(len(words_probs)), words_probs))
        # words_probs = list(zip(range(len(words_probs)), words_probs, words_probs2))
        from pprint import pprint

        pprint(words_probs[:50])


if __name__ == "__main__":
    main()

