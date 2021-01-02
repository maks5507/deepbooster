#
# Created by maks5507 (me@maksimeremeev.com)
#

from typing import Callable
import torch
import json


class Dataset(torch.utils.data.Dataset):
    def __init__(self, chunk_path: str, transformer: Callable = None, apply_transformer_to_label: bool = False):
        self.chunk_path = chunk_path
        self.transformer = transformer
        self.apply_transformer_to_label = apply_transformer_to_label

        self.chunk = []
        with open(self.chunk_path, 'r') as f:
            for line in f:
                self.chunk += [json.loads(line)]

    def __getitem__(self, index: int):
        input = self.chunk[index]['input']
        label = self.chunk[index]['label']

        if self.transformer is not None:
            input = self.transformer(input)
            if self.apply_transformer_to_label:
                label = self.transformer(label)

        return input, label

    def __len__(self):
        return len(self.chunk)
