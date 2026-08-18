# dataloader.py
import torch
from tokenizer import TokenizerBPE
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class PixarDataset(Dataset):
    def __init__(
        self,
        lines: list[str],
        context_len: int,
        begin_id: int,
        end_id: int,
        pad_id: int,
        tokenizer: TokenizerBPE,
    ) -> None:
        super().__init__()
        self.lines = lines
        self.context_len = context_len
        self.tokenizer = tokenizer
        self.begin_id = begin_id
        self.end_id = end_id

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx) -> torch.Tensor:
        tokens = self.tokenizer.encode(self.lines[idx])
        max_content_len = self.context_len - 2
        tokens = tokens[:max_content_len]
        seq = [self.begin_id] + tokens + [self.end_id]
        return torch.tensor(seq)


def collate_fn(batch: list[torch.Tensor], pad_id: int) -> torch.Tensor:
    return pad_sequence(batch, batch_first=True, padding_value=pad_id)
