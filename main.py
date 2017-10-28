import numpy as np
from networks import StackedAutoEncoder
from networks import ForwardClassifier
from networks import Lstm
from utils import Utils as utils

#----------------common-variables----------------
e_values = np.load("./data/e_records.npy")
e_classes = np.load("./data/e_classes.npy")
t_values = np.load("./data/t_records.npy")
t_classes = np.load("./data/t_classes.npy")

training_frac = 0.8
apply_reduction = True

attributes_num = len(e_values[0][0])
classes_num = len(e_classes[0][0])

#---------------------LSTM-----------------------
print("---------------------LSTM-----------------------")
lstm_e_values = np.concatenate((e_values, e_classes), axis=2)
lstm_t_values = np.concatenate((t_values, t_classes), axis=2)

max_sequence_length = 5

lstm_e_values, lstm_e_classes, lstm_e_lengths = utils.rnn_shift_padding(e_values, e_classes, max_sequence_length)
lstm_t_values, lstm_t_classes, lstm_t_lengths = utils.rnn_shift_padding(t_values, t_classes, max_sequence_length)

if(apply_reduction):
    selection = np.random.choice(len(lstm_e_values), min(len(lstm_e_values), len(lstm_t_values)), replace=False)
    lstm_e_values, lstm_e_classes, lstm_e_lengths = lstm_e_values[selection], lstm_e_classes[selection], lstm_e_lengths[selection]

lstm_values = np.concatenate((lstm_e_values, lstm_t_values))
lstm_classes = np.concatenate((lstm_e_classes, lstm_t_classes))
lstm_lengths = np.concatenate((lstm_e_lengths, lstm_t_lengths))

cost_mask = utils.get_cost_mask(lstm_classes) / 10

input_size = len(lstm_values[0][0])
output_size = len(lstm_classes[0][0])
lstm = Lstm(max_sequence_length=max_sequence_length, input_size=input_size, state_size=50, output_size=output_size, 
            loss_function='weighted-sparse-softmax-cross-entropy', initialization_function='xavier', 
            optimization_function='gradient-descent', learning_rate=0.05, batch_size=32, epoch=1, cost_mask=cost_mask)


lstm_train, lstm_test = utils.generate_rnn_train_test(lstm_values, lstm_classes, lstm_lengths, training_frac)
print("Training LSTM...")
lstm.train(lstm_train[0], lstm_train[1], lstm_train[2])
print("Error on training set:")
lstm.test(lstm_train[0], lstm_train[1], lstm_train[2])
print("Error on test set:")
lstm.test(lstm_test[0], lstm_test[1], lstm_test[2])

#---------------------SDAE-----------------------
print("---------------------SDAE-----------------------")
sdae_e_values = np.reshape(e_values,(-1,attributes_num))
sdae_t_values = np.reshape(t_values,(-1,attributes_num))

if(apply_reduction):
    selection = np.random.choice(len(sdae_e_values), min(len(sdae_e_values), len(sdae_t_values)), replace=False)
    sdae_e_values = sdae_e_values[selection]

sdae_values = np.concatenate((sdae_e_values, sdae_t_values))

'''
sdae = StackedAutoEncoder(dims=[25], activations=['relu'], decoding_activations=['sigmoid'], noise=['mask-0.7'],
                        epoch=[3000], loss=['cross-entropy'], lr=0.05, batch_size=100, print_step=200)
'''
sdae = StackedAutoEncoder(dims=[50, 50, 25], activations=['tanh', 'tanh', 'relu'], decoding_activations=['sigmoid', 'sigmoid', 'sigmoid'], 
                        noise=['mask-0.7','gaussian','gaussian'], epoch=[100, 100, 100], loss=['cross-entropy','rmse','rmse'], lr=0.01, 
                        batch_size=100, print_step=200)

sdae_train, sdae_test = utils.generate_sdae_train_test(sdae_values, training_frac)
print("Training SDAE...")
sdae.train(sdae_values)
print("Finetuning SDAE...")
sdae.finetune(sdae_train)
#model.test(test_X, 10, threshold=0.1)

#-----------------feed-forward-------------------
print("-----------------feed-forward-------------------")
classifier_e_values = np.reshape(e_values,(-1, attributes_num))
classifier_t_values = np.reshape(t_values,(-1, attributes_num))
classifier_e_classes = np.reshape(e_classes,(-1, classes_num))
classifier_t_classes = np.reshape(t_classes,(-1, classes_num))

classifier_e_values, classifier_e_classes = utils.homogenize(classifier_e_values, classifier_e_classes, 0.3)
classifier_t_values, classifier_t_classes = utils.homogenize(classifier_t_values, classifier_t_classes, 0.3)

if(apply_reduction):
    selection = np.random.choice(len(classifier_e_values), min(len(classifier_e_values), len(classifier_t_values)), replace=False)
    classifier_e_values, classifier_e_classes = classifier_e_values[selection], classifier_e_classes[selection]

classifier_values = np.concatenate((classifier_e_values, classifier_t_values))
classifier_classes = np.concatenate((classifier_e_classes, classifier_t_classes))

classifier = ForwardClassifier(input_size=attributes_num, output_size=classes_num, dims=[80,20], activation_functions=['relu','relu'], 
                            output_activation_function='softmax', loss_function='rmse', optimization_function='adam', epoch=1000,
                            learning_rate=0.007, batch_size=100, print_step=200)
classifier_train, classifier_test = utils.generate_classifier_train_test(classifier_values, classifier_classes, training_frac)
print("Training Classifier...")
classifier.train(classifier_train[0], classifier_train[1])
print("Error on training set:")
classifier.test(classifier_train[0], classifier_train[1])
print("Error on test set:")
classifier.test(classifier_test[0], classifier_test[1])

#--------------sdae-feed-forward-----------------
print("--------------sdae-feed-forward-----------------")
sdae_classifier_e_values = sdae.encode(np.reshape(e_values,(-1, attributes_num)))
sdae_classifier_t_values = sdae.encode(np.reshape(t_values,(-1, attributes_num)))
sdae_classifier_e_classes = np.reshape(e_classes,(-1, classes_num))
sdae_classifier_t_classes = np.reshape(t_classes,(-1, classes_num))

classifier_e_values, classifier_e_classes = utils.homogenize(classifier_e_values, classifier_e_classes, 0.3)
classifier_t_values, classifier_t_classes = utils.homogenize(classifier_t_values, classifier_t_classes, 0.3)

if(apply_reduction):
    selection = np.random.choice(len(sdae_classifier_e_values), min(len(sdae_classifier_e_values), len(sdae_classifier_t_values)), replace=False)
    sdae_classifier_e_values, sdae_classifier_e_classes = sdae_classifier_e_values[selection], sdae_classifier_e_classes[selection]

sdae_classifier_values = np.concatenate((sdae_e_values, sdae_t_values))
sdae_classifier_classes = np.concatenate((sdae_classifier_e_classes, sdae_classifier_t_classes))

sdae_classifier = ForwardClassifier(input_size=attributes_num, output_size=classes_num, dims=[80,20], activation_functions=['relu','relu'], 
                            output_activation_function='softmax', loss_function='rmse', optimization_function='adam', epoch=1000,
                            learning_rate=0.007, batch_size=100, print_step=200)

sdae_classifier_train, sdae_classifier_test = utils.generate_classifier_train_test(sdae_classifier_values, sdae_classifier_classes, training_frac)
print("Training SDAE Classifier...")
sdae_classifier.train(sdae_classifier_train[0], sdae_classifier_train[1])
print("Error on training set:")
sdae_classifier.test(sdae_classifier_train[0], sdae_classifier_train[1])
print("Error on test set:")
sdae_classifier.test(sdae_classifier_test[0], sdae_classifier_test[1])