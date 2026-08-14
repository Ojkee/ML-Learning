A measure of how much one probability distribution $Q$ differs from another distribution $P$.

It can be interpreted as the **extra information (or surprise) needed when using $Q$ to represent data that actually follows $P$**.

For discrete probability distributions:

$$
D_{KL}(P\parallel Q)
=

\sum_x P(x)\log\frac{P(x)}{Q(x)}
$$

where:

* $P$ - the **true/reference distribution**
* $Q$ - the **approximating/predicted distribution**

### Example

Suppose the true distribution is:

$$
\begin{aligned}
P(A) &= 0.7 \\
P(B) &= 0.3
\end{aligned}
$$

and our model predicts:

$$
\begin{aligned}
Q(A) &= 0.4 \\
Q(B) &= 0.6
\end{aligned}
$$

Then:

$$
\begin{aligned}
D_{KL}(P\parallel Q)
&=
P(A)\log\frac{P(A)}{Q(A)}
+
P(B)\log\frac{P(B)}{Q(B)}
\\
&=
0.7\log\frac{0.7}{0.4}
+
0.3\log\frac{0.3}{0.6}
\end{aligned}
$$

Using natural logarithms:

$$
D_{KL}(P\parallel Q)\approx0.184
$$

The smaller the KL divergence, the more similar $Q$ is to $P$.
