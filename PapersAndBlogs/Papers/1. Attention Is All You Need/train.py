from functools import partial
from typing import Literal

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from config import TrainConfig
from dataloader import collate_fn
from model import TransformerSeq2Seq


def lr(step: int, cfg: TrainConfig):
    lhs = (step + 1) ** -0.5
    rhs = (step + 1) * cfg.lr_warmup_steps**-1.5
    return cfg.lr_factor * min(lhs, rhs)


@torch.no_grad()
def evaluate(model, loader, device, pad_id, max_batches=None):
    model.eval()
    total_loss, n_batches = 0.0, 0
    for i, (src, dec_in, dec_tgt, src_mask, tgt_mask) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        src, dec_in, dec_tgt = src.to(device), dec_in.to(device), dec_tgt.to(device)
        src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)
        logits = model(src, dec_in, src_pad_mask=src_mask, tgt_pad_mask=tgt_mask)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), dec_tgt.view(-1), ignore_index=pad_id
        )
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


def run_training(
    model: "TransformerSeq2Seq",
    cfg: "TrainConfig",
    train_dataset: Dataset,
    valid_dataset: Dataset,
    pad_id: int,
    device: Literal["cuda", "cpu"],
    max_batch_per_epoch: int | None = None,
):
    model = model.to(device)

    collate = partial(collate_fn, pad_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=1.0,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr(step, cfg),
    )

    train_losses = []
    valid_losses = []
    global_step = 0

    total_batches = min(
        len(train_loader),
        (max_batch_per_epoch if max_batch_per_epoch is not None else len(train_loader)),
    )

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0

        progress = tqdm(
            train_loader,
            total=total_batches,
            desc=f"Epoch {epoch + 1}/{cfg.epochs}",
        )

        for i, batch in enumerate(progress):
            if max_batch_per_epoch is not None and i >= max_batch_per_epoch:
                break

            src, dec_in, dec_tgt, src_pad_mask, tgt_pad_mask = batch
            src = src.to(device)
            dec_in = dec_in.to(device)
            dec_tgt = dec_tgt.to(device)
            src_mask = src_pad_mask.to(device)
            tgt_mask = tgt_pad_mask.to(device)

            logits = model(src, dec_in, src_mask, tgt_mask)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                dec_tgt.view(-1),
                ignore_index=pad_id,
                label_smoothing=cfg.label_smoothing_eps,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_loss += loss.item()
            n_batches += 1

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append((epoch, avg_train_loss))

        if epoch % cfg.eval_every_epochs == 0:
            avg_valid_loss = evaluate(
                model,
                valid_loader,
                device,
                pad_id,
                max_batches=cfg.eval_max_batches,
            )

            valid_losses.append((epoch, avg_valid_loss))

            print(
                f"epoch {epoch}: "
                f"train_loss={avg_train_loss:.4f} "
                f"valid_loss={avg_valid_loss:.4f}"
            )

    return train_losses, valid_losses
