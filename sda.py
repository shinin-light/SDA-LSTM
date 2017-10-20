import numpy as np
from deepautoencoder import StackedAutoEncoder
from classifier import ForwardClassifier

values = np.load("./data/records.npy")
classes = np.load("./data/classes.npy")

training_frac = 0.8
idx = np.random.rand(values.shape[0]) < training_frac

print("Creating SDA...")

train_X = values[idx]
test_X = values[~idx]
train_Y = classes[idx]
test_Y = classes[~idx]

model = StackedAutoEncoder(dims=[50, 50, 25], activations=['tanh', 'tanh', 'relu'], decoding_activations=['sigmoid', 'sigmoid', 'sigmoid'], noise=['mask-0.7','gaussian','gaussian'],
                        epoch=[3000, 3000, 3000], loss=['cross-entropy','rmse','rmse'], lr=0.01, batch_size=100, print_step=200)
'''
model = StackedAutoEncoder(dims=[25], activations=['relu'], decoding_activations=['sigmoid'], noise=['mask-0.7'],
                        epoch=[3000], loss=['cross-entropy'], lr=0.01, batch_size=100, print_step=200)
'''

print("Creating pre-SDA Classifier...")
classifier1 = ForwardClassifier(dims=[20], activations=['tanh'], output_activation='softmax', epoch=[10000], loss='softmax-cross-entropy', lr=0.01, batch_size=100, print_step=200)
classifier1.fit(train_X, train_Y)

print("Error on training set:")
classifier1.test(train_X, train_Y)
print("Error on test set:")
classifier1.test(test_X, test_Y)



print("Fitting SDA...")
model.fit(values)
#model.test(test_X, 0)
model.finetune(train_X)
#model.test(test_X, 10, threshold=0.1)
coded_train_X = model.encode(train_X)
coded_test_X = model.encode(test_X)

print("Creating post-SDA Classifier...")
classifier1 = ForwardClassifier(dims=[20], activations=['tanh'], output_activation='softmax', epoch=[10000], loss='softmax-cross-entropy', lr=0.01, batch_size=100, print_step=200)
classifier1.fit(coded_train_X, train_Y)

print("Error on training set:")
classifier1.test(coded_train_X, train_Y)
print("Error on test set:")
classifier1.test(coded_test_X, test_Y)


#test_X_ = model.encode(test_X)

print("Ending.")
