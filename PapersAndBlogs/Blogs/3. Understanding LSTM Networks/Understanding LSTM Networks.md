# Link (2026-08-16)
https://colah.github.io/posts/2015-08-Understanding-LSTMs/

# Concept
The idea behind LSTMs is described in [[The Unreasonable Effectiveness of Recurrent Neural Networks]].

# Architecture
![[Pasted image 20260816204511.png]]
![[Pasted image 20260816204555.png]]
### The top line
![[Pasted image 20260816205525.png]]

Essential core of LSTM architecture. This line flows through entire chain.

First interaction is point-wise multiplication of sigmoid layer, the sigmoid is in range $0-1$, thus this might be interpreted as 'how much should this line forget about past' (`forget gate layer`).
Second interaction is influence of current input to memory (sigmoid layer in this interaction is called `input gate layer`).
Last interaction is influence of memory to current input.

# Variants

![[Pasted image 20260816210834.png]]![[Pasted image 20260816210926.png|700]]![[Pasted image 20260816211049.png]]
