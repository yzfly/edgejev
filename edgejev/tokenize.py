"""tokenizers 的薄封装：各后端共用，运行时不需要 transformers。"""
from typing import List


class Encoder:
    def __init__(self, tok, ids_map):
        self.tok = tok
        for k, v in ids_map.items():
            setattr(self, k, v)

    @classmethod
    def from_file(cls, path, ids_map):
        from tokenizers import Tokenizer
        return cls(Tokenizer.from_file(path), ids_map)

    def ids(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False).ids
