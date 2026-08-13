from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    name: str
    d_vocab: int
    context_len: int
    d_k: int
    d_v: int
    d_model: int
    h: int
    N: int
    dropout_rate: float


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    beta1: float
    beta2: float
    eps: float
    lr_factor: float
    lr_warmup_steps: int
    label_smoothing_eps: float
    eval_every_epochs: int
    eval_max_batches: int
