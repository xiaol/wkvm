"""RWKV World tokenizer as a slow HF tokenizer (byte-level greedy trie).

Loads ``rwkv_vocab_v20230424.txt`` (one line per token: ``<id> <python
literal> <byte length>``). Token strings are latin-1 views of the byte
sequences so the HF slow-tokenizer plumbing (tokenize -> ids, ids -> string)
works unchanged; ``convert_tokens_to_string`` re-decodes the bytes as UTF-8.
Id 0 is the model's end-of-text token (not in the vocab file); it decodes to
the empty string and is the tokenizer's eos.
"""

from __future__ import annotations

import os

from transformers import PreTrainedTokenizer

VOCAB_FILE = "rwkv_vocab_v20230424.txt"
EOS_ID = 0
EOS_TOKEN = "<|endoftext|>"


class _Trie:
    __slots__ = ("children", "token_id")

    def __init__(self) -> None:
        self.children: dict[int, _Trie] = {}
        self.token_id: int | None = None


class RwkvWorldTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": VOCAB_FILE}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file: str, **kwargs) -> None:
        self.vocab_file = vocab_file
        self.id_to_bytes: dict[int, bytes] = {}
        with open(vocab_file, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                idx = int(line[: line.index(" ")])
                literal = line[line.index(" ") + 1 : line.rindex(" ")]
                tok = eval(literal)  # noqa: S307 - the vocab file is a trusted python-literal list
                self.id_to_bytes[idx] = tok if isinstance(tok, bytes) else tok.encode("utf-8")
        self.bytes_to_id = {b: i for i, b in self.id_to_bytes.items()}
        self.root = _Trie()
        for i, b in self.id_to_bytes.items():
            node = self.root
            for byte in b:
                node = node.children.setdefault(byte, _Trie())
            node.token_id = i
        kwargs.setdefault("eos_token", EOS_TOKEN)
        super().__init__(**kwargs)

    # -- HF slow-tokenizer contract ------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return max(self.id_to_bytes) + 1

    def get_vocab(self) -> dict[str, int]:
        vocab = {b.decode("latin-1"): i for i, b in self.id_to_bytes.items()}
        vocab[EOS_TOKEN] = EOS_ID
        return vocab

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        data = text.encode("utf-8")
        out: list[str] = []
        pos = 0
        while pos < len(data):
            node, best_id, best_len, i = self.root, None, 0, pos
            while i < len(data) and data[i] in node.children:
                node = node.children[data[i]]
                i += 1
                if node.token_id is not None:
                    best_id, best_len = node.token_id, i - pos
            if best_id is None:
                raise ValueError(f"byte {data[pos]!r} at {pos} not in vocab")
            out.append(self.id_to_bytes[best_id].decode("latin-1"))
            pos += best_len
        return out

    def _convert_token_to_id(self, token: str) -> int:
        if token == EOS_TOKEN:
            return EOS_ID
        return self.bytes_to_id[token.encode("latin-1")]

    def _convert_id_to_token(self, index: int) -> str:
        if index == EOS_ID:
            return EOS_TOKEN
        return self.id_to_bytes[index].decode("latin-1")

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return b"".join(t.encode("latin-1") for t in tokens if t != EOS_TOKEN).decode("utf-8", errors="replace")

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        import shutil

        dst = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILE)
        if os.path.abspath(self.vocab_file) != os.path.abspath(dst):
            shutil.copyfile(self.vocab_file, dst)
        return (dst,)
