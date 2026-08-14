# Beam Search

Search algorithm that keeps the $k$ most probable **partial sequences (beams)** at each decoding step.

Unlike greedy search, which keeps only the most probable token at each step, beam search keeps multiple promising candidates. This allows it to recover from an initially suboptimal token choice.

### Example

For $k=2$, let's say the model with $d_{vocab}=N$ predicts the following probabilities at the first step:

$$
\begin{aligned}
P(word_{1,1}) &= 0.5 \\
P(word_{1,2}) &= 0.01 \\
P(word_{1,3}) &= 0.45 \\
&\vdots \\
P(word_{1,N}) &= \ldots
\end{aligned}
$$

We keep the $k=2$ most probable tokens:

$$
\begin{aligned}
P(word_{1,1}) &= 0.5 \\
P(word_{1,3}) &= 0.45
\end{aligned}
$$

Greedy search would keep only $word_{1,1}$.

### Second iteration

Now the model predicts the next token **conditioned on each of the current beams**.

For $word_{1,1}$:

$$
\begin{aligned}
P(word_{2,1}\mid word_{1,1}) &= 0.2 \\
P(word_{2,2}\mid word_{1,1}) &= 0.05 \\
P(word_{2,3}\mid word_{1,1}) &= 0.64 \\
&\vdots \\
P(word_{2,N}\mid word_{1,1}) &= \ldots
\end{aligned}
$$

For $word_{1,3}$:

$$
\begin{aligned}
P(word_{2,1}\mid word_{1,3}) &= 0.9 \\
P(word_{2,2}\mid word_{1,3}) &= 0.02 \\
P(word_{2,3}\mid word_{1,3}) &= 0.03 \\
&\vdots \\
P(word_{2,N}\mid word_{1,3}) &= \ldots
\end{aligned}
$$

We now have $k \times d_{vocab}$ possible sequences. We calculate their probabilities and keep the $k$ best ones.

For example:

$$
\begin{aligned}
P(word_{1,1}, word_{2,1})
&= P(word_{1,1})P(word_{2,1}\mid word_{1,1}) \\
&= 0.5\cdot0.2 = 0.10 \\[6pt]
P(word_{1,1}, word_{2,3})
&= P(word_{1,1})P(word_{2,3}\mid word_{1,1}) \\
&= 0.5\cdot0.64 = 0.32 \\[6pt]
P(word_{1,3}, word_{2,1})
&= P(word_{1,3})P(word_{2,1}\mid word_{1,3}) \\
&= 0.45\cdot0.9 = 0.405 \\[6pt]
P(word_{1,3}, word_{2,3})
&= P(word_{1,3})P(word_{2,3}\mid word_{1,3}) \\
&= 0.45\cdot0.03 = 0.0135
\end{aligned}
$$

The two most probable sequences are therefore:

$$
\begin{aligned}
P(word_{1,3}, word_{2,1}) &= 0.405 \\
P(word_{1,1}, word_{2,3}) &= 0.32
\end{aligned}
$$

These become the new **beams**.

For longer sequences, we repeat the same process: expand each of the $k$ current beams, calculate the probabilities of all resulting sequences, and keep only the $k$ most probable ones. We continue until the end-of-sequence (`EOS`) token is generated.

### Greedy vs. Beam Search

In this example, greedy search would choose:

$$
word_{1,1} \rightarrow word_{2,3}
$$

with probability:

$$
0.5\cdot0.64=0.32
$$

Beam search keeps:

$$
word_{1,3} \rightarrow word_{2,1}
$$

with probability:

$$
0.45\cdot0.9=0.405
$$

as well as the greedy sequence.

The important difference is that **greedy search commits to one choice at every step, while beam search keeps $k$ promising alternatives**.

### Log Probabilities

For numerical stability, we usually work with log probabilities.

Instead of multiplying probabilities:

$$
P(x_1,\ldots,x_n)
=
\prod_{i=1}^{n}P(x_i\mid x_1,\ldots,x_{i-1})
$$

we sum their logarithms:

$$
\log P(x_1,\ldots,x_n)
=
\sum_{i=1}^{n}\log P(x_i\mid x_1,\ldots,x_{i-1})
$$

because:

$$
\log(ab)=\log a+\log b
$$

This avoids numerical underflow when multiplying many small probabilities.