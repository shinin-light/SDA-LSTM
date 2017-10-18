import numpy as np
from deepautoencoder import StackedAutoEncoder

values = np.load("./data/records.npy")

training_frac = 0.8
idx = np.random.rand(values.shape[0]) < training_frac

print("Creating SDA...")

train_X = values[idx]
test_X = values[~idx]

model = StackedAutoEncoder(dims=[100, 50, 20], activations=['relu', 'relu', 'relu'], noise='mask-0.1',
                        epoch=[3000, 3000, 3000], loss='rmse', lr=0.007, batch_size=500, print_step=200)
model.fit(train_X)
test_X_ = model.transform(test_X)

print("Ending.")
