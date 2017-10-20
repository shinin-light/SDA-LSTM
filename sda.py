import numpy as np
from deepautoencoder import StackedAutoEncoder
from classifier import ForwardClassifier

e_values = np.load("./data/e_records.npy")
e_classes = np.load("./data/e_classes.npy")
t_values = np.load("./data/t_records.npy")
t_classes = np.load("./data/t_classes.npy")

e_values = np.delete(e_values, 0, 1)
t_values = np.delete(t_values, 0, 1)

#SDAE VARIABLES
#Reduction of the bigger dataset to the dimension of the smaller one
selection = np.random.choice(len(e_values), min(len(t_values),len(e_values)), replace=False)
sdae_e_values = e_values[selection]

values = np.concatenate((sdae_e_values, t_values))

training_frac = 0.8
idx = np.random.rand(values.shape[0]) < training_frac

sdae_train_X = values[idx]

#CLASSIFIER VARIABLES
e_valid_class_idx = np.where(e_classes.any(axis=1))[0]
t_valid_class_idx = np.where(t_classes.any(axis=1))[0]

classifier_e_values = e_values[e_valid_class_idx]
classifier_e_classes = e_classes[e_valid_class_idx]
classifier_t_values = t_values[t_valid_class_idx]
classifier_t_classes = t_classes[t_valid_class_idx]

#Reduction of the bigger dataset to the dimension of the smaller one
selection = np.random.choice(len(classifier_e_values), min(len(classifier_t_values),len(classifier_e_values)), replace=False)
classifier_e_values = classifier_e_values[selection]
classifier_e_classes = classifier_e_classes[selection]
selection = np.random.choice(len(classifier_t_values), min(len(classifier_t_values),len(classifier_e_values)), replace=False)
classifier_t_values = classifier_t_values[selection]
classifier_t_classes = classifier_t_classes[selection]

classifier_values = np.concatenate((classifier_e_values, classifier_t_values))
classifier_classes = np.concatenate((classifier_e_classes, classifier_t_classes))

training_frac = 0.8
cidx = np.random.rand(classifier_values.shape[0]) < training_frac

classifier_train_X = classifier_values[cidx]
classifier_train_Y = classifier_classes[cidx]
classifier_test_X = classifier_values[~cidx]
classifier_test_Y = classifier_classes[~cidx]

print("Creating SDA...")

#train_valid_class_idx = np.where(train_Y.any(axis=1))[0]
#test_valid_class_idx = np.where(test_Y.any(axis=1))[0]

'''
model = StackedAutoEncoder(dims=[50, 50, 25], activations=['tanh', 'tanh', 'relu'], decoding_activations=['sigmoid', 'sigmoid', 'sigmoid'], noise=['mask-0.7','gaussian','gaussian'],
                        epoch=[3000, 3000, 3000], loss=['cross-entropy','rmse','rmse'], lr=0.01, batch_size=100, print_step=200)
'''

model = StackedAutoEncoder(dims=[25], activations=['relu'], decoding_activations=['sigmoid'], noise=['mask-0.7'],
                        epoch=[3000], loss=['cross-entropy'], lr=0.05, batch_size=100, print_step=200)


print("Creating pre-SDA Classifier...")
classifier1 = ForwardClassifier(dims=[80,20], activations=['relu','relu'], output_activation='softmax', epoch=10000, loss='rmse', lr=0.007, batch_size=100, print_step=200)
classifier1.fit(classifier_train_X, classifier_train_Y)

print("Error on training set:")
classifier1.test(classifier_train_X, classifier_train_Y)
print("Error on test set:")
classifier1.test(classifier_test_X, classifier_test_Y)

print("Fitting SDA...")
model.fit(values)
model.finetune(sdae_train_X)
#model.test(test_X, 10, threshold=0.1)
coded_train_X = model.encode(classifier_train_X)
coded_test_X = model.encode(classifier_test_X)

print("Creating post-SDA Classifier...")
classifier1 = ForwardClassifier(dims=[80,20], activations=['relu','relu'], output_activation='softmax', epoch=10000, loss='rmse', lr=0.007, batch_size=100, print_step=200)
classifier1.fit(coded_train_X, classifier_train_Y)

print("Error on training set:")
classifier1.test(coded_train_X, classifier_train_Y)
print("Error on test set:")
classifier1.test(coded_test_X, classifier_test_Y)


#test_X_ = model.encode(test_X)

print("Ending.")
