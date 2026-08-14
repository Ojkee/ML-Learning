# Link (2026-08-14)
https://nlp.seas.harvard.edu/annotated-transformer/

# Architecture
This paper focuses on implementing Transformer, so the architecture is more-less the same as in the [[Attention Is All You Need]].

# Training 
- Track stats
```python
class TrainState:
    step: int = 0
    accum_step: int = 0 
    samples: int = 0 
    tokens: int = 0
```
```python
# epoch loop

train_state.step += 1
train_state.samples += batch.src.shape[0]
train_state.tokens += batch.ntokens

# rest of the loop
```




