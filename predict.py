import numpy as np
from fuzzy_knn import FuzzyKNN

sample = np.random.rand(1, 128)
model = FuzzyKNN(k=3)

print("Prediction:", model.predict(sample))
