import numpy as np
from networks import StackedAutoEncoder
from networks import ForwardClassifier
from networks import Lstm
from networks import Svm
from utils import Utils as utils
from sklearn import utils as skutils
import os

training_frac = 0.8
apply_reduction = True

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

attributes_num = len(e_values[0][0])
classes_num = len(e_classes[0][0])

max_sequence_length = np.max([len(e_values[0]),len(t_values[0])])
e_values, e_classes, e_lengths = utils.rnn_shift_padding(e_values, e_classes, max_sequence_length)
t_values, t_classes, t_lengths = utils.rnn_shift_padding(t_values, t_classes, max_sequence_length)

#if(apply_reduction):
#    selection = np.random.choice(len(svm_e_values), min(len(svm_e_values), len(svmt_values)), replace=False)
#    svm_e_values, svm_e_classes = svm_e_values[selection], svm_e_classes[selection]

rnn_values = np.concatenate((e_values, t_values))
rnn_classes = np.concatenate((e_classes, t_classes))
rnn_lengths = np.concatenate((e_lengths, t_lengths))

rnn_train, rnn_test = utils.generate_rnn_train_test(rnn_values, rnn_classes, rnn_lengths, training_frac)


flat_values = np.reshape(rnn_values,(-1, attributes_num))
flat_classes = np.reshape(rnn_classes,(-1, classes_num))
flat_values, flat_classes = utils.homogenize(flat_values, flat_classes, 0.3)

flat_train, flat_test = utils.generate_flat_train_test(flat_values, flat_classes, training_frac)


cost_mask = utils.get_cost_mask(rnn_classes)
cost_mask /= np.mean(cost_mask)
#weights = skutils.compute_class_weight(class_weight='balanced', classes=np.array(range(10)), y=np.argmax(flat_classes, 1))

#---------------------SVM------------------------
print("---------------------SVM------------------------")

svm = Svm(cost_mask=cost_mask)

print("Training SVM...")
svm.train(flat_train[0], flat_train[1])
print("Error on training set:")
svm.test(flat_train[0], flat_train[1])
print("Error on test set:")
svm.test(flat_test[0], flat_test[1])

#---------------------LSTM-----------------------
print("---------------------LSTM-----------------------")

input_size = len(rnn_values[0][0])
lstm = Lstm(scope_name='basic-lstm', max_sequence_length=max_sequence_length, input_size=input_size, state_size=50, 
            output_size=classes_num, loss_function='weighted-sparse-softmax-cross-entropy', initialization_function='xavier',
            optimization_function='gradient-descent', learning_rate=0.05, learning_rate_decay='fraction', batch_size=32, 
            epochs=10, cost_mask=cost_mask, noise='gaussian')

print("Training LSTM...")
lstm.train(rnn_train[0], rnn_train[1], rnn_train[2])
print("Error on training set:")
lstm.test(rnn_train[0], rnn_train[1], rnn_train[2])
print("Error on test set:")
lstm.test(rnn_test[0], rnn_test[1], rnn_test[2])

#---------------------SDAE-----------------------
print("---------------------SDAE-----------------------")

sdae = StackedAutoEncoder(scope_name='three-layers-sdae', input_size=attributes_num, dims=[150, 100, 50], optimization_function='adam',
                        encoding_functions=['tanh', 'tanh', 'tanh'], decoding_functions=['sigmoid', 'tanh', 'tanh'],
                        noise=['mask-0.7','gaussian','gaussian'], epochs=10, loss_functions=['rmse','rmse','rmse'], 
                        learning_rate=0.01, batch_size=128)

print("Training SDAE...")
sdae.train(flat_values)
print("Finetuning SDAE...")
sdae.finetune(flat_train[0])
#sdae.test(flat_test[0], 1, threshold=0.1)

#------------------CLASSIFIER--------------------
print("------------------CLASSIFIER--------------------")

classifier = ForwardClassifier(scope_name='basic-forward', input_size=attributes_num, output_size=classes_num, dims=[80,20], 
                            activation_functions=['relu','relu'], output_activation_function='softmax', loss_function='rmse', 
                            optimization_function='adam', epochs=10, learning_rate=0.05, batch_size=128)

print("Training CLASSIFIER...")
classifier.train(flat_train[0], flat_train[1])
print("Error on training set:")
classifier.test(flat_train[0], flat_train[1])
print("Error on test set:")
classifier.test(flat_test[0], flat_test[1])

#----------------SDAE-CLASSIFIER-----------------
print("----------------SDAE-CLASSIFIER-----------------")

sdae_classifier_train = sdae.encode(flat_train[0])
sdae_classifier_test = sdae.encode(flat_test[0])
input_size = len(sdae_classifier_train[0])
sdae_classifier = ForwardClassifier(scope_name='sdae-forward', input_size=input_size, output_size=classes_num, dims=[80,20], 
                            activation_functions=['relu','relu'], output_activation_function='softmax', loss_function='rmse', 
                            optimization_function='adam', epochs=10, learning_rate=0.05, batch_size=128)

print("Training SDAE-CLASSIFIER...")
sdae_classifier.train(sdae_classifier_train, flat_train[1])
print("Error on training set:")
sdae_classifier.test(sdae_classifier_train, flat_train[1])
print("Error on test set:")
sdae_classifier.test(sdae_classifier_test, flat_test[1])

#-------------------SDAE-LSTM--------------------
print("-------------------SDAE-LSTM--------------------")

sdae_lstm_train = sdae.timeseries_encode(rnn_train[0])
sdae_lstm_test = sdae.timeseries_encode(rnn_test[0])
input_size = len(sdae_lstm_train[0][0])
sdae_lstm = Lstm(scope_name='sdae-lstm', max_sequence_length=max_sequence_length, input_size=input_size, state_size=50, 
            output_size=classes_num, loss_function='weighted-sparse-softmax-cross-entropy', initialization_function='xavier', 
            optimization_function='gradient-descent', learning_rate=0.05, learning_rate_decay='fraction', batch_size=32, 
            epochs=10, cost_mask=cost_mask)

print("Training SDAE-LSTM...")
sdae_lstm.train(sdae_lstm_train, rnn_train[1], rnn_train[2])
print("Error on training set:")
sdae_lstm.test(sdae_lstm_train, rnn_train[1], rnn_train[2])
print("Error on test set:")
sdae_lstm.test(sdae_lstm_test, rnn_test[1], rnn_test[2])

#--------------------SDAE-SVM--------------------
print("--------------------SDAE-SVM--------------------")

sdae_svm_train = sdae.encode(flat_train[0])
sdae_svm_test = sdae.encode(flat_test[0])
sdae_svm = Svm(cost_mask=cost_mask)

print("Training SDAE-SVM...")
sdae_svm.train(sdae_svm_train, flat_train[1])
print("Error on training set:")
sdae_svm.test(sdae_svm_train, flat_train[1])
print("Error on test set:")
sdae_svm.test(sdae_svm_test, flat_test[1])
