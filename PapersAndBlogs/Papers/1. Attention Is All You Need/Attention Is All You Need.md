# Architecture
![[Pasted image 20260805204142.png]]
### Encoder
- Composed of $N = 6$ identical layers
- Each layer is build of Multi-Head Self-Attention & Fully Connected Feed-Forward
- Residual Connection around sub-layers
- Layer Normalization after layer
- Each sub-layer, embedding layer produce output of $d_\text{model}$ (Compatible with Residual Connections)
$$
\text{LayerNorm}(x + \text{Sub-layer}(x))
$$
### Decoder
- Also stack of $N=6$ layers
- Multi-Head combines previous output to output of the encoder
- Additionally, there is one more Multi-Head sub-layer with **mask** to prevent including future tokens to attention

# Attention
![[Pasted image 20260806191741.png]]

### Scaled Dot-Product
Attention mechanism is basically mapping with separate Linear layers of input to Query $Q$,  Key $K$ and Value $V$. Each Linear layer has weights of shape $d_\text{model} \times d_k$,  where $d_k$ is the head size (dimensionality of each query/key/value vector).  For a sequence of length (context length) $n$, this gives $Q, K \in \mathbb{R}^{d_k \times d_k}$.  The $Q$ matrix is multiplied with $K^T$, resulting in an $d_k \times d_k$ matrix of attention scores,  and then the mask is applied. It results in $-\infty$ in the upper-right triangular part.

$$
M \in \mathbb{R}^{d_k \times d_k}, \qquad
M = 
\begin{bmatrix}
x_{11} & -\infty & -\infty & -\infty \\
x_{21} & x_{22} & -\infty & -\infty \\
\vdots & \vdots & \ddots & \vdots \\
x_{d_k1} & x_{d_k2} & \dots & x_{d_kd_k}
\end{bmatrix}
$$
This matrix is scaled with factor of $\frac{1}{\sqrt{d_k}}$ (with large dimensionality $d_k$,  dot products grow large in magnitude, pushing the softmax function into regions with small  gradients, thus scaling prevents that; note $d_k = h$ in this notation). Then, the probability  of each score is calculated with softmax, so the $-\infty$s become $0$. Finally, the  probabilities are matmulled with the $V$ matrix.
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### Multi-Head Attention
Its just $h$ attention heads, each head contains own set of Linear layers for $Q$, $K$ and $V$. After performing Scaled Dot-Product in every head, outputs are concatenated and Linearly projected with Linear Layer.
$$
\text{MultiHead}(Q, K, V)=
\text{Concat}(head_1, \dots, head_h)W^O
$$
where 
$$
head_i = 
\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$
$$
W_i^Q \in \mathbb{R}^{d_\text{model}\times d_k}
$$
$$
W_i^K \in \mathbb{R}^{d_\text{model}\times d_k}
$$
$$
W_i^V \in \mathbb{R}^{d_\text{model}\times d_v}
$$
$$
W^O \in \mathbb{R}^{hd_v\times d_\text{model}}
$$
Usually $d_v = d_k$ . Authors used:
- $h=8$
- $d_\text{model} = 512$ 
- $d_k=d_v=d_\text{model}/h=64$.

### Position-wise Feed-Forward Networks
After each attention, there is Feed-Forward network with hidden layer of size $d_\text{hidden}$ (authors used $d_\text{model} * 4$), $ReLU$ activation and projection layer of size $d_\text{model}$.

### Embeddings and Softmax

Input tokens are converted into embeddings with size $d_\text{model}$ via an embedding matrix $W_e \in \mathbb{R}^{d_\text{vocab} \times d_\text{model}}$. At the very end of the decoder block, there is a final Linear layer with softmax activation to predict the next token, so the output dimension of that layer is $d_\text{vocab}$. The authors used the **same weight matrix** for both the embedding layer and this final projection layer (weight tying):
$$
\text{logits} = h \, W_e^T, \qquad h \in \mathbb{R}^{n \times d_\text{model}}, \quad 
W_e^T \in \mathbb{R}^{d_\text{model} \times d_\text{vocab}}
$$
Embeddings are scaled with factor of $\sqrt{d_\text{model}}$ and there is Positional Encoding added.
$$
x = W_e[\text{token\_id}] \cdot \sqrt{d_\text{model}} + \text{PE}
$$
### Positional Encoding
The model adds positional encodings to the scaled input embeddings to inject information about 
the relative or absolute position of tokens in the sequence. The authors used sine and cosine 
functions of different frequencies:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_\text{model}}}\right)
$$
$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_\text{model}}}\right)
$$
##### Example
With 4 embedding dimensions per token, $i \in \{0, 1\}$ (since $2i$ and $2i+1$ must cover  indices $0,1,2,3$).
$$
PE_{(pos, 0)} = \sin\left(\frac{pos}{10000^{0/4}}\right) = \sin(pos)
$$
$$
PE_{(pos, 1)} = \cos\left(\frac{pos}{10000^{0/4}}\right) = \cos(pos)
$$
$$
PE_{(pos, 2)} = \sin\left(\frac{pos}{10000^{2/4}}\right) = \sin(pos / 100)
$$
$$
PE_{(pos, 3)} = \cos\left(\frac{pos}{10000^{2/4}}\right) = \cos(pos / 100)
$$

For $pos = 0$:
$$
PE_{(0,:)} = [\sin(0), \cos(0), \sin(0), \cos(0)] = [0, 1, 0, 1]
$$
For $pos = 1$:
$$
PE_{(1,:)} = [\sin(1), \cos(1), \sin(0.01), \cos(0.01)] \approx [0.841, 0.540, 0.010, 0.9999]
$$
For $pos = 2$:
$$
PE_{(2,:)} = [\sin(2), \cos(2), \sin(0.02), \cos(0.02)] \approx [0.909, -0.416, 0.020, 0.9998]
$$

# Training
### Optimizer
- Adam 
- $\beta_1 = 0.9$, $\beta_2 = 0.98$  and $\epsilon = 10^{-9}$ 
- learning rate with scheduler: 
$$
\text{lrate} =
	d_{model}^{-0.5} \enspace \cdot 
	\min(
		\text{step\_num}^{-0.5}, 
		\text{step\_num} \enspace \cdot \text{warmup\_steps}^{-1.5}
	)
$$
- $\text{warmup\_steps} = 4000$

### Regularization
- Dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized
- Dropout applied to the sums of the embeddings and the positional encodings in the Encoder and the Decoder
- $P_\text{drop} = 0.1$
- Label smoothing of value $\epsilon_\text{ls}=0.1$
