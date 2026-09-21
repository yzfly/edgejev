"""tokenizers 的薄封装：各后端共用，运行时不需要 transformers。"""
import re
from typing import List

#: 把用户文本里的 `<|name|>` 改写成 `<¦name¦>`，这样调用方无论写什么都造不出分隔符
#: token。kev 的布局靠 <opt>/</opt> 标定选项边界，边界可伪造就等于读出位可被劫持。
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


class Encoder:
    def __init__(self, tok, ids_map):
        self.tok = tok
        self.special = []
        for k, v in ids_map.items():
            setattr(self, k, v)

    @classmethod
    def from_file(cls, path, ids_map):
        from tokenizers import Tokenizer
        return cls(Tokenizer.from_file(path), ids_map)

    def ids(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def user_ids(self, text: str) -> List[int]:
        """调用方提供的文本，分隔符不可伪造。"""
        return self.ids(_SPECIAL_RE.sub(r"<¦\1¦>", text))
