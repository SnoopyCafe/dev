
import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # TODO: Compute and return softmax(x)

    return np.exp(x) / np.sum(np.exp(x), axis=0)

def softmax1(x):
    """Compute softmax values for each sets of scores in x."""
    # TODO: Compute and return softmax(x)
    logits = []
    for y in x:
        logits.append(np.exp(y) / np.sum(np.exp(x)))

    return logits

logits = [3.0, 1.0, 0.2]
sm = softmax(logits)
print (np.mean(np.square(sm)))
print(sm)

sm1 = softmax1(logits)
print (np.mean(np.square(sm1)))
print(sm1)
