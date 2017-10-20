import numpy as np
from deepautoencoder import StackedAutoEncoder
from classifier import ForwardClassifier

e_values = np.load("./data/e_records.npy")
e_classes = np.load("./data/e_classes.npy")
t_values = np.load("./data/t_records.npy")
t_classes = np.load("./data/t_classes.npy")

sdae_e_values = np.delete(e_values, 0, 1)
sdae_t_values = np.delete(t_values, 0, 1)

#Reduction of the bigger dataset to the dimension of the smaller one
selection = np.random.choice(len(sdae_e_values), len(sdae_t_values), replace=False)
sdae_e_values = sdae_e_values[selection]
e_classes = e_classes[selection]

values = np.concatenate((sdae_e_values, sdae_t_values))
classes = np.concatenate((e_classes, t_classes))

training_frac = 0.8
idx = np.random.rand(values.shape[0]) < training_frac

print("Creating SDA...")

train_X = values[idx]
test_X = values[~idx]
train_Y = classes[idx]
test_Y = classes[~idx]

train_valid_class_idx = np.where(train_Y.any(axis=1))[0]
test_valid_class_idx = np.where(test_Y.any(axis=1))[0]

'''
model = StackedAutoEncoder(dims=[50, 50, 25], activations=['tanh', 'tanh', 'relu'], decoding_activations=['sigmoid', 'sigmoid', 'sigmoid'], noise=['mask-0.7','gaussian','gaussian'],
                        epoch=[3000, 3000, 3000], loss=['cross-entropy','rmse','rmse'], lr=0.01, batch_size=100, print_step=200)
'''

model = StackedAutoEncoder(dims=[25], activations=['relu'], decoding_activations=['sigmoid'], noise=['mask-0.7'],
                        epoch=[3000], loss=['cross-entropy'], lr=0.01, batch_size=100, print_step=200)


print("Creating pre-SDA Classifier...")
classifier1 = ForwardClassifier(dims=[20], activations=['tanh'], output_activation='softmax', epoch=10000, loss='softmax-cross-entropy', lr=0.01, batch_size=100, print_step=200)
classifier1.fit(train_X[train_valid_class_idx], train_Y[train_valid_class_idx])

print("Error on training set:")
classifier1.test(train_X[train_valid_class_idx], train_Y[train_valid_class_idx])
print("Error on test set:")
classifier1.test(test_X[test_valid_class_idx], test_Y[test_valid_class_idx])

print("Fitting SDA...")
model.fit(values)
#model.test(test_X, 0)
model.finetune(train_X)
#model.test(test_X, 10, threshold=0.1)
coded_train_X = model.encode(train_X)
coded_test_X = model.encode(test_X)

print("Creating pre-SDA Classifier...")
classifier1 = ForwardClassifier(dims=[20], activations=['tanh'], output_activation='softmax', epoch=10000, loss='softmax-cross-entropy', lr=0.01, batch_size=100, print_step=200)
classifier1.fit(train_X[train_valid_class_idx], train_Y[train_valid_class_idx])

print("Error on training set:")
classifier1.test(train_X[train_valid_class_idx], train_Y[train_valid_class_idx])
print("Error on test set:")
classifier1.test(test_X[test_valid_class_idx], test_Y[test_valid_class_idx])


#test_X_ = model.encode(test_X)

print("Ending.")
