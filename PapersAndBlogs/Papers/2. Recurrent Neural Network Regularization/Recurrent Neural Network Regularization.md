# Concept
Authors point on poor performance of dropout in RNNs.

Dropout 'inject' some noise to the forward pass, that helps with generalization. The problem arises when RNNs use noisy output as input in next recurrent iteration, amplifies the noise. 

Paper focuses on another approach to applying dropout to RNNs, and in consequence, reducing overfitting.

# Architecture
Authors used Graves el a. (2013) LSTM architecture as base model for experiments.
![[Pasted image 20260819120303.png|630]]

![[Pasted image 20260819120327.png|631]]

Parenthesis are vector concatenations here for BLAS/GPU optimalization.

# Dropout Connections
![[Pasted image 20260819121630.png]]

![[Pasted image 20260819132120.png|626]]

Don't want want to add to much noise to history, because of that the dropout is applied between layers, and not between long memory states.

