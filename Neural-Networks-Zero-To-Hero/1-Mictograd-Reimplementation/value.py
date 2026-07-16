from __future__ import annotations

import math
import random

from typing import Callable

type back_fn = Callable[[], None]
type other_t = Value | int | float


class Value:
    def __init__(self, data: float, _children: tuple[Value, ...] = ()) -> None:
        self.data = data
        self.grad = 0.0

        self._children: set[Value] = set(_children)

        self._backward = lambda: None

    @classmethod
    def of(cls, other) -> Value:
        match other:
            case Value():
                return other
            case int() | float():
                return Value(other)
        raise NotImplementedError()

    @classmethod
    def random_norm(cls) -> Value:
        return cls(random.normalvariate())

    def __repr__(self) -> str:
        return f"V({self.data:.3f})"

    def __add__(self, other: other_t) -> Value:
        other = Value.of(other)
        out = Value(self.data + other.data, (self, other))

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other) -> Value:
        return self + other

    def __sub__(self, other) -> Value:
        return self + -other

    def __rsub__(self, other) -> Value:
        return self - other

    def __neg__(self) -> Value:
        return -1 * self

    def __truediv__(self, other: other_t) -> Value:
        return self * (other ** (-1))

    def __mul__(self, other: other_t) -> Value:
        other = Value.of(other)
        out = Value(self.data * other.data, (self, other))

        def _backward() -> None:
            self.grad += out.grad * other.data
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: other_t) -> Value:
        return self * other

    def __pow__(self, other: other_t) -> Value:
        if not isinstance(other, int | float):
            raise NotImplementedError("not supported yet")

        out = Value(self.data**other, (self,))

        def _backward() -> None:
            self.grad += out.grad * other * (self.data ** (other - 1))

        out._backward = _backward
        return out

    def exp(self) -> Value:
        out = Value(math.exp(self.data), (self,))

        def _backward() -> None:
            self.grad += out.grad * out.data

        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(max(self.data, 0), (self,))

        def _backward() -> None:
            self.grad += out.grad if out.data > 0 else 0

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        _2exp = (2 * self).exp()
        out = (_2exp - 1) / (_2exp + 1)

        def _backward() -> None:
            self.grad += out.grad * (1 - out.data**2)

        out._backward = _backward
        return out

    def backward(self) -> None:
        visited: set[Value] = set()
        topology: list[Value] = []

        def _topology(node: Value) -> None:
            if node in visited:
                return
            visited.add(node)
            for child in node._children:
                _topology(child)
            topology.append(node)

        self.grad = 1.0
        _topology(self)
        for node in reversed(topology):
            node._backward()

    def zero_grad(self) -> None:
        self.grad = 0
        for child in self._children:
            child.zero_grad()
