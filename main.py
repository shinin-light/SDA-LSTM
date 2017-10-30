import numpy as np
from networks import StackedAutoEncoder
from networks import ForwardClassifier
from networks import Lstm
from utils import Utils as utils
import os

#--------------------folders---------------------

folders_file = './folders'
folders  = open(folders_file, 'r').read().split('\n')
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

#----------------common-variables----------------
e_values = np.load("./data/e_records.npy")
e_classes = np.load("./data/e_classes.npy")
t_values = np.load("./data/t_records.npy")
t_classes = np.load("./data/t_classes.npy")

e_values = np.concatenate((e_values, e_classes), axis=2)
t_values = np.concatenate((t_values, t_classes), axis=2)

training_frac = 0.8
apply_reduction = True

attributes_num = len(e_values[0][0])
classes_num = len(e_classes[0][0])

#---------------------LSTM-----------------------
print("---------------------LSTM-----------------------")

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
lstm = Lstm(scope_name='basic-lstm', max_sequence_length=max_sequence_length, input_size=input_size, state_size=50, 
            output_size=output_size, loss_function='weighted-sparse-softmax-cross-entropy', initialization_function='xavier',
            optimization_function='gradient-descent', learning_rate=0.05, learning_rate_decay='fraction', batch_size=32, 
            epoch=10, cost_mask=cost_mask, noise='gaussian')

lstm_train, lstm_test = utils.generate_rnn_train_test(lstm_values, lstm_classes, lstm_lengths, training_frac)
print("Training LSTM...")
lstm.train(lstm_train[0], lstm_train[1], lstm_train[2])
print("Error on training set:")
lstm.test(lstm_train[0], lstm_train[1], lstm_train[2])
print("Error on test set:")
lstm.test(lstm_test[0], lstm_test[1], lstm_test[2])

#---------------------SDAE-----------------------
print("---------------------SDAE-----------------------")
sdae_e_values = np.reshape(e_values,(-1, attributes_num))
sdae_t_values = np.reshape(t_values,(-1, attributes_num))
sdae_e_classes = np.reshape(e_classes,(-1, classes_num))
sdae_t_classes = np.reshape(t_classes,(-1, classes_num))

sdae_e_values, sdae_e_classes = utils.homogenize(sdae_e_values, sdae_e_classes, 0.3)
sdae_t_values, sdae_t_classes = utils.homogenize(sdae_t_values, sdae_t_classes, 0.3)

if(apply_reduction):
    selection = np.random.choice(len(sdae_e_values), min(len(sdae_e_values), len(sdae_t_values)), replace=False)
    sdae_e_values = sdae_e_values[selection]

sdae_values = np.concatenate((sdae_e_values, sdae_t_values))
sdae_classes = np.concatenate((sdae_e_values, sdae_t_values))


sdae = StackedAutoEncoder(scope_name='basic-sdae', input_size=attributes_num, dims=[100], encoding_functions=['relu'], decoding_functions=['sigmoid'], 
                        noise=['mask-0.5'], epoch=[10], loss_functions=['rmse'], optimization_function='gradient-descent', learning_rate=0.05, 
                        batch_size=128)
'''
sdae = StackedAutoEncoder(input_size=attributes_num, dims=[150, 100, 50], encoding_functions=['tanh', 'tanh', 'relu'], 
                        decoding_functions=['sigmoid', 'sigmoid', 'sigmoid'], noise=['mask-0.7','gaussian','gaussian'], epoch=[10, 10, 10], 
                        loss_functions=['sigmoid-cross-entropy','rmse','rmse'], optimization_function='adam', learning_rate=0.01, batch_size=128)
'''

sdae_train, sdae_test = utils.generate_sdae_train_test(sdae_values, training_frac)
print("Training SDAE...")
sdae.train(sdae_values)
print("Finetuning SDAE...")
sdae.finetune(sdae_train)
#sdae.test(sdae_train, 10, threshold=0.1)

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

classifier = ForwardClassifier(scope_name='basic-forward', input_size=attributes_num, output_size=classes_num, dims=[80,20], 
                            activation_functions=['relu','relu'], output_activation_function='softmax', loss_function='rmse', 
                            optimization_function='adam', epoch=10, learning_rate=0.05, batch_size=128)

classifier_train, classifier_test = utils.generate_classifier_train_test(classifier_values, classifier_classes, training_frac)
print("Training Classifier...")
classifier.train(classifier_train[0], classifier_train[1])
print("Error on training set:")
classifier.test(classifier_train[0], classifier_train[1])
print("Error on test set:")
classifier.test(classifier_test[0], classifier_test[1])

#---------------sdae-feed-forward----------------
print("---------------sdae-feed-forward----------------")
sdae_classifier_e_values = sdae.encode(np.reshape(e_values,(-1, attributes_num)))
sdae_classifier_t_values = sdae.encode(np.reshape(t_values,(-1, attributes_num)))
sdae_classifier_e_classes = np.reshape(e_classes,(-1, classes_num))
sdae_classifier_t_classes = np.reshape(t_classes,(-1, classes_num))

sdae_classifier_e_values, sdae_classifier_e_classes = utils.homogenize(sdae_classifier_e_values, sdae_classifier_e_classes, 0.3)
sdae_classifier_t_values, sdae_classifier_t_classes = utils.homogenize(sdae_classifier_t_values, sdae_classifier_t_classes, 0.3)

if(apply_reduction):
    selection = np.random.choice(len(sdae_classifier_e_values), min(len(sdae_classifier_e_values), len(sdae_classifier_t_values)), replace=False)
    sdae_classifier_e_values, sdae_classifier_e_classes = sdae_classifier_e_values[selection], sdae_classifier_e_classes[selection]

sdae_classifier_values = np.concatenate((sdae_classifier_e_values, sdae_classifier_t_values))
sdae_classifier_classes = np.concatenate((sdae_classifier_e_classes, sdae_classifier_t_classes))

input_size = len(sdae_classifier_values[0])
sdae_classifier = ForwardClassifier(scope_name='sdae-forward', input_size=input_size, output_size=classes_num, dims=[80,20], 
                            activation_functions=['relu','relu'], output_activation_function='softmax', loss_function='rmse', 
                            optimization_function='adam', epoch=10, learning_rate=0.05, batch_size=128)

sdae_classifier_train, sdae_classifier_test = utils.generate_classifier_train_test(sdae_classifier_values, sdae_classifier_classes, training_frac)
print("Training SDAE Classifier...")
sdae_classifier.train(sdae_classifier_train[0], sdae_classifier_train[1])
print("Error on training set:")
sdae_classifier.test(sdae_classifier_train[0], sdae_classifier_train[1])
print("Error on test set:")

#-------------------SDAE-LSTM--------------------
print("-------------------SDAE-LSTM--------------------")
sdae_lstm_e_values = np.array(sdae.timeseries_encode(e_values))
sdae_lstm_t_values = np.array(sdae.timeseries_encode(t_values))
sdae_lstm_e_classes = e_classes
sdae_lstm_t_classes = t_classes

max_sequence_length = 5

sdae_lstm_e_values, sdae_lstm_e_classes, sdae_lstm_e_lengths = utils.rnn_shift_padding(sdae_lstm_e_values, sdae_lstm_e_classes, max_sequence_length)
sdae_lstm_t_values, sdae_lstm_t_classes, sdae_lstm_t_lengths = utils.rnn_shift_padding(sdae_lstm_t_values, sdae_lstm_t_classes, max_sequence_length)

if(apply_reduction):
    selection = np.random.choice(len(sdae_lstm_e_values), min(len(sdae_lstm_e_values), len(sdae_lstm_t_values)), replace=False)
    sdae_lstm_e_values, sdae_lstm_e_classes, sdae_lstm_e_lengths = sdae_lstm_e_values[selection], sdae_lstm_e_classes[selection], sdae_lstm_e_lengths[selection]

sdae_lstm_values = np.concatenate((sdae_lstm_e_values, sdae_lstm_t_values))
sdae_lstm_classes = np.concatenate((sdae_lstm_e_classes, sdae_lstm_t_classes))
sdae_lstm_lengths = np.concatenate((sdae_lstm_e_lengths, sdae_lstm_t_lengths))

cost_mask = utils.get_cost_mask(sdae_lstm_classes) / 10

input_size = len(sdae_lstm_values[0][0])
output_size = len(sdae_lstm_classes[0][0])
sdae_lstm = Lstm(scope_name='sdae-lstm', max_sequence_length=max_sequence_length, input_size=input_size, state_size=50, 
            output_size=output_size, loss_function='weighted-sparse-softmax-cross-entropy', initialization_function='xavier', 
            optimization_function='gradient-descent', learning_rate=0.05, learning_rate_decay='fraction', batch_size=32, 
            epoch=10, cost_mask=cost_mask, noise='gaussian')

sdae_lstm_train, sdae_lstm_test = utils.generate_rnn_train_test(sdae_lstm_values, sdae_lstm_classes, sdae_lstm_lengths, training_frac)
print("Training LSTM...")
sdae_lstm.train(sdae_lstm_train[0],sdae_lstm_train[1], sdae_lstm_train[2])
print("Error on training set:")
sdae_lstm.test(sdae_lstm_train[0], sdae_lstm_train[1], sdae_lstm_train[2])
print("Error on test set:")
sdae_lstm.test(sdae_lstm_test[0], sdae_lstm_test[1], sdae_lstm_test[2])