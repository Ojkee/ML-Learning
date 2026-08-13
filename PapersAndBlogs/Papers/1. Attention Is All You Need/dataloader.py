import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(
        self, en_path, pl_path, sp_model, start_id, end_id, pad_id, context_len
    ):
        with open(en_path, encoding="utf-8") as en_file:
            self.en_lines = [line.strip() for line in en_file]
        with open(pl_path, encoding="utf-8") as pl_file:
            self.pl_lines = [line.strip() for line in pl_file]
        self.sp = sp_model
        self.start_id = start_id
        self.end_id = end_id
        self.pad_id = pad_id
        self.context_len = context_len

    def __len__(self):
        return len(self.en_lines)

    def __getitem__(self, idx):
        src = self.sp.encode(self.en_lines[idx], out_type=int)[: self.context_len]
        tgt = self.sp.encode(self.pl_lines[idx], out_type=int)[: self.context_len - 2]
        tgt = [self.start_id] + tgt + [self.end_id]
        return torch.tensor(src), torch.tensor(tgt[:-1]), torch.tensor(tgt[1:])


def collate_fn(pad_id, batch):
    src, dec_in, dec_tgt = zip(*batch)

    src = pad_sequence(
        src,  # type: ignore
        batch_first=True,
        padding_value=pad_id,
    )

    dec_in = pad_sequence(
        dec_in,  # type: ignore
        batch_first=True,
        padding_value=pad_id,
    )

    dec_tgt = pad_sequence(
        dec_tgt,  # type: ignore
        batch_first=True,
        padding_value=pad_id,
    )

    src_padding_mask = src.eq(pad_id)
    tgt_padding_mask = dec_in.eq(pad_id)

    return (
        src,
        dec_in,
        dec_tgt,
        src_padding_mask,
        tgt_padding_mask,
    )
