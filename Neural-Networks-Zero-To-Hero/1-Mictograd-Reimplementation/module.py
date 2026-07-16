from itertools import pairwise
from typing import Collection

from value import Value

type input_t = int | float | Value


class Module:
    @property
    def params(self) -> list[Value]:
        return []

    def zero_grad(self) -> None:
        for param in self.params:
            param.zero_grad()


class Neuron(Module):
    def __init__(self, n_inputs: int) -> None:
        self._ws = [Value.random_norm() for _ in range(n_inputs)]
        self._b = Value.random_norm()

    def __call__(self, xs: Collection[input_t]) -> Value:
        acc = sum((wi * xi for wi, xi in zip(self._ws, xs)), self._b)
        return acc.tanh()

    @property
    def params(self) -> list[Value]:
        return self._ws + [self._b]


class Layer(Module):
    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        super().__init__()
        self._neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, xs: Collection[input_t]) -> list[Value]:
        return [n(xs) for n in self._neurons]

    @property
    def params(self) -> list[Value]:
        _params = []
        for neuron in self._neurons:
            _params.extend(neuron.params)
        return _params

    def __repr__(self) -> str:
        fmt = ", ".join(str(p) for p in self.params)
        return f"[{fmt}]"


class NeuralNetwork(Module):
    def __init__(self, n_inputs: int, hidden_dims: list[int], n_outputs: int) -> None:
        super().__init__()
        sizes = [n_inputs] + hidden_dims + [n_outputs]
        self._layers = [Layer(*n) for n in pairwise(sizes)]

    def __call__(self, xs: Collection[input_t]) -> list[Value]:
        for layer in self._layers:
            xs = layer(xs)
        return xs  # type: ignore

    @property
    def params(self) -> list[Value]:
        _params = []
        for layer in self._layers:
            _params.extend(layer.params)
        return _params

    def __repr__(self) -> str:
        def fmt_row(i: int, layer: Layer) -> str:
            return f"{i}: [{repr(layer):>30}]"

        return "\n".join(fmt_row(i, layer) for i, layer in enumerate(self._layers))

    def __len__(self) -> int:
        return len(self.params)

    def train(self, *, learning_rate: float) -> None:
        for param in self.params:
            param.data -= param.grad * learning_rate
