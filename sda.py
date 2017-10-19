import numpy as np
from deepautoencoder import StackedAutoEncoder

values = np.load("./data/records.npy")

training_frac = 0.8
idx = np.random.rand(values.shape[0]) < training_frac

print("Creating SDA...")

train_X = values[idx]
test_X = values[~idx]

model = StackedAutoEncoder(dims=[50, 30, 15], activations=['tanh', 'tanh', 'tanh'], decoding_activations=['sigmoid', 'tanh', 'tanh'], noise=['mask-0.3','gaussian','gaussian'],
                        epoch=[3000, 3000, 3000], loss=['cross-entropy','rmse','rmse'], lr=0.01, batch_size=100, print_step=200)
model.fit(train_X)
model.finetune(train_X)
test_X_ = model.transform(test_X)

print("Ending.")
