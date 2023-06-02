import numpy as np
from hmmlearn.hmm import GaussianHMM

model = GaussianHMM(n_components=2)

data = np.array([ 
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [3, 4, 5, 6, 7],
    [4, 5, 6, 7, 8],
    [5, 6, 7, 8, 9]])

model.fit(data)

next_word = model.predict(np.array([[1], [2], [3]]))
print(next_word)
