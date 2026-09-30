# Concept
Main focus is neural nets ability to generalize. The idea to make it easier for nets to achieve, is by adding some noise to the weights while training. This way, weights keep less information than the output vector of training example.

# Minimum Description Length Principle 
More complex models can fit data easier, but at some point, more complexity means, that the model fits too well on training data and performs poorly on new data.

Minimum Description Length Principle minimizes the cost of describing the model and the difference between output and target value.

Authors suggest to think of this in such way, let's assume we have the sender and the reciever. Sender have the input vector and the model, reciever have only the input vector. The goal is to send the model minimizing the cost of transmission and the error of the output. Each weight costs $-\log p$, where $p$ is the each weight.

# Coding the data misfits
Authors assume a well-trained model should produce errors close to zero more frequently than larger errors, with no systematic bias in either direction. This assumption motivates choosing a zero-mean Gaussian with standard deviation $\sigma_j$ as the coding distribution for residuals.

![[Pasted image 20260921211636.png|629]]

Given that, probability of error ($d^c_j - y^c_j$)  comes to this:
$$
p(d^c_j - y^c_j) = 
t\frac1{\sqrt{2\pi}\sigma_j}\exp{
\bigg[
	\frac{-(d^c_j - y^c_j)^2}{2\sigma^2_j}
\bigg]}
$$

taking the $\log$ (calculating bits - Shannon) and summing over all the example cases, we obtain:
$$
C_{\text{data-misfit}}=kN+\frac{N}{2}\log
\bigg[
	\frac1N\sum_{c}(d_j^c-y^c_j)^2
\bigg]
$$
where $k$ is constant that depends on $t$.

Minimizing this equation comes down to minimizing sum of squared errors, because $\log$ monotonic function and left part is constant, which is basically just minimizing the MSE.
That means, that coding the residuals is done by calculating MSE.

# A simple method of coding the weights
The idea for coding weights is the same as coding the misfits, but instead of $d^c_j - y^c_j$, we calculate for each weight $w_{ij}$. Making same steps:
$$
-\log(p(w_{ij}))=
-\log t 
+\log \sqrt{2\pi}
+\log \sigma_w
+ \frac{w_{ij}}{2\sigma^2_w}
$$
since 3 first parts are constant, we only care about the last one, which is proportional to just weight itself
$$
\frac{w_{ij}}{2\sigma^2_w} \propto w_{ij}
$$
and all of it comes down to
$$
\frac1{2\sigma^2_w}\sum_{ij}w^2_{ij}
$$
giving the final sum $C$ 
$$
C = C_{\text{data-misfit}} + C_{\text{weights}} =
\sum_j\frac1{2\sigma^2_j}
\sum_c(d^c_j-y^c_j)^2 +
\frac1{2\sigma^2_w}\sum_{ij}w^2_{ij}
$$
Double sum results from two dimentions we have to consider. 
- $j$ is the index of output neuron
- $c$ is the index of example

This kind of modelling doesn't take to account important fact, some weights may be represented with lower precision (bigger $t$), and some require higher (lower $t$).

# Noisy weights
Adding the noise as zero-mean Gaussian noise is a way to limit the amount of information in a number.

Instead of using weights as value, we can use Gaussian with mean of this value and variance. Weight with greater variance may have greater difference to the original value (greater variance of data misfit), but it doesn't need that much precision, thus, making it cheaper to send.

Every number needs some way to be coded to be send, every coding schema implies some probability distribution which numbers are cheap to be coded, which are expensive. Sender and reciever need to decide on such probability $P$. Assume, we learned the model and we get the posterior distribution $Q$ that tells us which weights can be coded cheaply, and which are more expensive. 

To calculate how much $P$ differs from $Q$, authors used KL-divergence. 
$$
G(P, Q) = \int Q(w)\log\frac{Q(w)}{P(w)}dw
$$

### Bits-back
