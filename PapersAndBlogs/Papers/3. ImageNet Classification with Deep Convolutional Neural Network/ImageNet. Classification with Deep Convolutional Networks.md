# Concept
Authors presented results for much deeper CNN architecture trained on much bigger dataset with more categories to classify. They applied modern (at time) regularization techniques.

# Architecture
- 5 Convolutional Layers
- 3 Fully-Connected 
- ReLU activations, used, because they are faster than $\tanh$s (require less iterations to achieve similar error rate, especially valuable for training on large datasets)

### Training on Multiple GPUs
The architecture was big enough to fit dataset of greater size and complexity, but the issue was to fit them into single GPU. Fortunately the GPUs supported cross-GPU well, they could read each other memory directly.

Authors placed half of the kernel maps (channels) of each layer on the first GPU and the remaining half on the second one. Normally, a convolutional layer needs to see every input channel to compute its output, but in this architecture some layers are restricted to only the feature maps residing on the same GPU, avoiding cross-GPU communication and making computation faster. Which layers use this restricted connectivity is a hyperparameter chosen via cross-validation.

### Local Response Normalization
$$
b^i_{x, y} = 
a^i_{x, y} / \Biggl(
k + \alpha \sum^{\min (N - 1, i + n / 2)}_{j=\max (0, i-n/2)} (a^j_{x,y})^2
\Biggl)^\beta
$$
Where:
- $a_{x, y}^i$ is activation of $i$'th kernel at position $(x, y)$
- $b$ is normalized activation
- $n$ is the number of adjacent kernel maps at same spatial position 
- $N$ is the total number of kernels in the layer

Hyperparameters:
- $k = 2$
- $n = 5$
- $\alpha = 10^{-4}$
- $\beta = 0.75$

This normalization is applied after ReLU activations for chosen layers.

### Overlapping Pooling
$s$ is the size of space between pooling units and $z$ length of pooling unit. 
$s = z$ means that pooling is evenly spaced and units does not overlap, but if $s \lt z$, then pooling overlaps.

Used in paper:
- $s = 2$
- $z=3$
Authors observed that overlapping pooling is slightly more difficult to overfit.

### Overall Architecture
![[Pasted image 20260827190721.png]]

- Response-normalization layers follow the first and second convolutional layers. 
- ReLU is applied to the output of every convolutional and fully-connected layer.

# Reducing Overfitting
### Data Augmentation
- Computed on CPU while the GPUs are running
- First form is obtained by applying translation and horizontal reflection to the random patches of size $224 \times 224$ from the original images which are of size  $256 \times 256$
- Second form is obtained by altering the intensities of the RGB channels. It is done by taking principal components calculated by PCA on image, those components are then multiplied by corresponding eigenvalues multiplied with random value drawn  from Gaussian with mean $=0$ and stddev $=0.1$.
$$
[
	\bf{p}_1,
	\bf{p}_2,
	\bf{p}_3
]
[
	\alpha_1\lambda_1, 
	\alpha_2\lambda_2, 
	\alpha_3\lambda_3
]^T
$$
- $\bf{p}_i$ and $\lambda_i$ are $i$th eigenvector and eigenvalue of the $3 \times 3$ covariance matrix of RGB pixel values
- $\alpha$ is random value, each $\alpha_i$ is drawn once for all the pixels of a particular training image until image is used again

### Dropout
- At train time, authors applied dropout with drop probability $=0.5$ to first two fully-connected layers
- At test time, authors multiplied outputs of every neuron by $0.5$

# Training
- SGD optimizer
- batch size $=128$
- momentum $=0.9$
- weight decay $=5\mathrm{e}{-4}$ 
- learning rate $\epsilon = 0.01$ at start, then drop with factor of $10$ when validation error stopped improving
$$
v_{i + 1} \coloneqq 
0.9 \cdot v_i 
- 0.0005 \cdot \epsilon \cdot w_i 
- \epsilon \cdot 
	\Bigl<
		 \frac{\partial L}{\partial w}|_{w_i} 
	\Bigl>_{D_i}
$$
$$ 
w_i \coloneqq w_i + v_{i + 1}
$$
- biases for second, fourth and fifth convolutional layers initialized with $1$
- biases fully-connected layers initialized with $1$
- biases for remaining layers initialized with $0$
