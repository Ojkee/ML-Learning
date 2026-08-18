import torch
from torch import nn


class PixarModel(nn.Module):
    def __init__(
        self,
        *,
        d_vocab: int,
        d_embedding: int,
        d_model: int,
        n_layers: int,
    ) -> None:
        super().__init__()
        self.d_hidden = d_model

        self.embeddings = nn.Embedding(d_vocab, d_embedding)
        self.lstms = nn.ModuleList(
            [
                VanillaLSTM(d_embedding if i == 0 else d_model, d_model)
                for i in range(n_layers)
            ]
        )
        self.projection = nn.Linear(d_model, d_vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        device = x.device
        emb = self.embeddings(x)

        h = [
            torch.zeros(B, self.d_hidden, device=device) for _ in range(len(self.lstms))
        ]
        C = [
            torch.zeros(B, self.d_hidden, device=device) for _ in range(len(self.lstms))
        ]

        all_logits = []
        for t in range(T):
            layer_input = emb[:, t]
            for i, lstm in enumerate(self.lstms):
                h[i], C[i] = lstm(layer_input, h[i], C[i])
                layer_input = h[i]

            all_logits.append(self.projection(h[-1]))

        return torch.stack(all_logits, dim=1)


class VanillaLSTM(nn.Module):
    def __init__(
        self,
        fan_in: int,
        d_hidden: int,
    ) -> None:
        super().__init__()
        self.forget_gate = nn.Sequential(
            nn.Linear(fan_in + d_hidden, d_hidden),
            nn.Sigmoid(),
        )

        self.input_gate = nn.Sequential(
            nn.Linear(fan_in + d_hidden, d_hidden),
            nn.Sigmoid(),
        )
        self.candidate_gate = nn.Sequential(
            nn.Linear(fan_in + d_hidden, d_hidden),
            nn.Tanh(),
        )

        self.output_gate = nn.Sequential(
            nn.Linear(fan_in + d_hidden, d_hidden),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        C: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.concat([x, h], dim=1)
        C = C * self.forget_gate(x)
        C = C + (self.input_gate(x) * self.candidate_gate(x))
        h = C.tanh() * self.output_gate(x)
        return h, C
