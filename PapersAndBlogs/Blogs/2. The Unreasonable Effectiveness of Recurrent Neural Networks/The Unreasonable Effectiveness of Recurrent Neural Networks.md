# Link (2026-08-15)
https://karpathy.github.io/2015/05/21/rnn-effectiveness/

# RNNs 
Neural Networks operate on fixed size input, therefore it is difficult for them to take input or generate output of various lengths like sentences (sequences).

Recurrent Neural Networks can operate on sequences.
![[Pasted image 20260815142006.png]]

They take the input and some current internal state and produce the next state. Its like the object with some variables, that combine result with some input and then, they are updated.

```python
class Foo:
	def __init__(self, start) -> None:
		self.state = start
		
	def learned_internal_f(self, x):
		return ...
		
	def other_transformation_f(self, x):
		return ...
	
	def final_transformation_f(self, x):
		return ...
	
	def next_value(self, input_value):
		self.state = (
			self.learned_internal_f(self.state)
			+ self.other_transformation_f(input_value)
		)
		return final_transformation_f(self.state)
```

```
If training vanilla neural nets is optimization over functions, training recurrent nets is optimization over programs.
```

# Interface
Vanilla RNN is rather simple.
```python
class VanillaRNN:
	# ...
	
	def step(self, x):
		self.h = tanh(
			dot_product(self.W_hh, self.h) 
			+ dot_product(self.W_xh, x)
		)
		y = dot_product(self.W_hy, self.h)
		return y

rnn = VanillaRNN()
y = rnn.step(x)
```

Where `self.W_hh`, `self.W_xy` and `self.W_yh` are weight matrices and `self.h` is previously mentioned **internal state**.

### Hidden state in math notation
$$
h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_{t})
$$
where $t$ is current time stamp and $\tanh$ is element-wise operation on vector.

### Stacking up
If done right, stacking models results in better performance. Its like stacking layers in single model or Attention blocks in Transformer.
```python
y1 = rnn1.step(x)
y2 = rnn2.step(y1)
```
`rnn1` and `rnn2` are two similar, but separate models. They don't know of each other.

### LSTM
In practice, much more common is LSTM, which is fancier variant of RNN. It uses more sophisticated internal transformation and has better backpropagation dynamics. 

# Generating flow
Assume, we want to train our model to say the word `meet`. Our strategy will be feeding the model character by character and the model would generate character by character output. 

If we had classic Neural Net:
- We feed the letter `m` and would get the next letter which is `e`
- Then we feed letter `e` and get the letter `e` again
- Lastly we feed the letter `e` again and get... 
how would model know whether we need the letter `t` this time, if we fed letter `e` and correctly got letter  `e` previously? The answer is this hidden state `h`, which is 'remembered' previous context.

# Training strategy
- Cross Entropy Loss
- Adam 
- 2 layer LSTM
- $d_{hidden}=512$
- 0.5 dropout after each layer
- backpropagation truncation with time of length of $100$ characters
- temperature below $1.0$
