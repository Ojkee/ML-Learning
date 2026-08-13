import os
from pathlib import Path

import sentencepiece as spm


class TokenizerBPE:
    def __init__(
        self, name: str, vocab_size: int, input_file_name: str | None = None
    ) -> None:
        self._name = name
        if not self.model_exists_in_current_dir():
            spm.SentencePieceTrainer.train(
                input=input_file_name or "./opus100/opus.en-pl-train.en_pl",
                input_format="text",
                model_prefix=self._name,
                model_type="bpe",
                vocab_size=vocab_size,
                normalization_rule_name="identity",
                remove_extra_whitespaces=False,
                input_sentence_size=200000000,
                max_sentence_length=10228,
                seed_sentencepiece_size=1000000,
                shuffle_input_sentence=True,
                character_coverage=0.99995,
                byte_fallback=True,
                split_digits=True,
                split_by_unicode_script=True,
                split_by_whitespace=True,
                split_by_number=True,
                max_sentencepiece_length=16,
                add_dummy_prefix=True,
                allow_whitespace_only_pieces=True,
                unk_id=0,
                bos_id=1,
                eos_id=2,
                pad_id=3,
                num_threads=os.cpu_count(),
            )
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(self.model_file_name)

    def model_exists_in_current_dir(self) -> bool:
        cwd = Path(__file__).parent
        _, _, files = next(os.walk(cwd))
        return self.model_file_name in files

    @property
    def model_file_name(self) -> str:
        return f"{self._name}.model"

    def encode(self, _input: str) -> list[int]:
        return self._sp.encode(_input)

    def decode(self, _input: list[int]) -> str:
        return self._sp.decode(_input)
