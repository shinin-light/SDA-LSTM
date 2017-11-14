import numpy as np
from networks import StackedAutoEncoder
from networks import ForwardClassifier
from networks import Lstm
from networks import Svm
from utils import Utils as utils
from sklearn import utils as skutils
from printer import Printer
import os

training_frac = 0.8
apply_reduction = True
class_is_yesno = True
class_reduction = 5

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

if not class_is_yesno and class_reduction is not None:
    e_classes = utils.reduce_classes_linear(e_classes, class_reduction)
    t_classes = utils.reduce_classes_linear(t_classes, class_reduction)

attributes_num = len(e_values[0][0])
classes_num = len(e_classes[0][0])

max_sequence_length = np.max([len(e_values[0]),len(t_values[0])])
e_values, e_classes, e_yesno, e_lengths = utils.rnn_shift_padding(e_values, e_classes, max_sequence_length)
t_values, t_classes, t_yesno, t_lengths = utils.rnn_shift_padding(t_values, t_classes, max_sequence_length)

#if(apply_reduction):
#    selection = np.random.choice(len(svm_e_values), min(len(svm_e_values), len(svmt_values)), replace=False)
#    svm_e_values, svm_e_classes = svm_e_values[selection], svm_e_classes[selection]

rnn_values = np.concatenate((e_values, t_values))
rnn_classes = np.concatenate((e_classes, t_classes))
rnn_yesno = np.concatenate((e_yesno, t_yesno))
rnn_lengths = np.concatenate((e_lengths, t_lengths))

flat_values = np.reshape(rnn_values,(-1, attributes_num))
flat_classes = np.reshape(rnn_classes,(-1, classes_num))
flat_yesno = np.reshape(rnn_yesno,(-1, 2))

if class_is_yesno :
    rnn_labels = rnn_yesno
    flat_labels = flat_yesno
else:
    rnn_labels = rnn_classes
    flat_labels = flat_classes

rnn_train, rnn_test = utils.generate_rnn_train_test(rnn_values, rnn_labels, rnn_lengths, training_frac)

flat_values, flat_labels = utils.homogenize(flat_values, flat_labels, 0) #just remove the invalid records.

flat_train, flat_test = utils.generate_flat_train_test(flat_values, flat_labels, training_frac)

sdae_train_values, sdae_train_labels = utils.homogenize(flat_train[0], flat_train[1], 1) #balancing sdae training values.
sdae_test_values, sdae_test_labels = utils.homogenize(flat_test[0], flat_test[1], 1) #balancing sdae test values.
sdae_train = np.concatenate((sdae_train_values, sdae_test_values))
sdae_finetuning_train = sdae_train_values

rnn_cost_mask = utils.get_cost_mask(rnn_labels)
rnn_cost_mask /= np.mean(rnn_cost_mask)

flat_cost_mask = utils.get_cost_mask(flat_labels)
flat_cost_mask /= np.mean(flat_cost_mask)

#weights = skutils.compute_class_weight(class_weight='balanced', classes=np.array(range(10)), y=np.argmax(flat_classes, 1))

alpha = 2

input_size = len(rnn_train[0][0][0])
output_size = len(rnn_train[1][0][0])
rnn_hidden_size = (len(rnn_train[0])) / (alpha * (input_size + output_size))
flat_hidden_size = (len(flat_train[0])) / (alpha * (input_size + output_size))

#--------------------printer---------------------

printer = Printer()

#---------------------SDAE-----------------------
printer.print("---------------------SDAE-----------------------")

sdae_output_size = 50
sdae = StackedAutoEncoder(scope_name='three-layers-sdae', input_size=input_size, dims=[150, 100, sdae_output_size], optimization_function='adam',
                        encoding_functions=['tanh', 'tanh', 'tanh'], decoding_functions=['linear', 'tanh', 'tanh'],
                        noise=['mask-0.7','gaussian','gaussian'], epochs=1, loss_functions=['sigmoid-cross-entropy','rmse','rmse'], 
                        learning_rate=0.01, learning_rate_decay='fraction', batch_size=128, printer=printer)

printer.print("Training SDAE...")
sdae.train(sdae_train)
printer.print("Finetuning SDAE...")
sdae.finetune(sdae_finetuning_train)
#sdae.test(flat_test[0], 1, threshold=0.1)

#---------------------SVM------------------------
printer.print("---------------------SVM------------------------")

svm = Svm(cost_mask=flat_cost_mask, output_size=output_size, printer=printer)

printer.print("Training SVM...")
svm.train(flat_train[0], flat_train[1])
printer.print("Error on training set:")
svm.test(flat_train[0], flat_train[1])
printer.print("Error on test set:")
svm.test(flat_test[0], flat_test[1])

#--------------------SDAE-SVM--------------------
printer.print("--------------------SDAE-SVM--------------------")

sdae_svm_train = sdae.encode(flat_train[0])
sdae_svm_test = sdae.encode(flat_test[0])
sdae_svm = Svm(cost_mask=flat_cost_mask, output_size=output_size, printer=printer)

printer.print("Training SDAE-SVM...")
sdae_svm.train(sdae_svm_train, flat_train[1])
printer.print("Error on training set:")
sdae_svm.test(sdae_svm_train, flat_train[1])
printer.print("Error on test set:")
sdae_svm.test(sdae_svm_test, flat_test[1])

#------------------CLASSIFIER--------------------
printer.print("------------------CLASSIFIER--------------------")

classifier = ForwardClassifier(scope_name='basic-forward', input_size=input_size, output_size=output_size, dims=[100], learning_rate_decay='fraction', 
                            activation_functions=['sigmoid'], loss_function='weighted-softmax-cross-entropy', cost_mask=flat_cost_mask,
                            metric_function='one-hot-accuracy', optimization_function='gradient-descent', epochs=1, learning_rate=0.1, batch_size=64, 
                            printer=printer)

printer.print("Training CLASSIFIER...")
classifier.train(flat_train[0], flat_train[1])
printer.print("Error on training set:")
classifier.test(flat_train[0], flat_train[1])
printer.print("Error on test set:")
classifier.test(flat_test[0], flat_test[1])

#----------------SDAE-CLASSIFIER-----------------
printer.print("----------------SDAE-CLASSIFIER-----------------")

sdae_classifier_train = sdae.encode(flat_train[0])
sdae_classifier_test = sdae.encode(flat_test[0])
sdae_classifier = ForwardClassifier(scope_name='sdae-forward', input_size=sdae_output_size, output_size=output_size, dims=[100,40], learning_rate_decay='fraction', 
                            activation_functions=['tanh','tanh'], loss_function='weighted-softmax-cross-entropy', cost_mask=flat_cost_mask,
                            metric_function='one-hot-accuracy', optimization_function='gradient-descent', epochs=1, learning_rate=0.05, batch_size=128, printer=printer)

printer.print("Training SDAE-CLASSIFIER...")
sdae_classifier.train(sdae_classifier_train, flat_train[1])
printer.print("Error on training set:")
sdae_classifier.test(sdae_classifier_train, flat_train[1])
printer.print("Error on test set:")
sdae_classifier.test(sdae_classifier_test, flat_test[1])

#---------------------LSTM-----------------------
printer.print("---------------------LSTM-----------------------")

lstm = Lstm(scope_name='basic-lstm', max_sequence_length=max_sequence_length, input_size=input_size, state_size=100, 
            output_size=output_size, loss_function='weighted-softmax-cross-entropy', initialization_function='xavier', metric_function='one-hot-accuracy',
            optimization_function='gradient-descent', learning_rate=0.1, learning_rate_decay='none', batch_size=32, 
            epochs=1, cost_mask=rnn_cost_mask, noise='none', printer=printer)

printer.print("Training LSTM...")
lstm.train(rnn_train[0], rnn_train[1], rnn_train[2])
printer.print("Error on training set:")
lstm.test(rnn_train[0], rnn_train[1], rnn_train[2])
printer.print("Error on test set:")
lstm.test(rnn_test[0], rnn_test[1], rnn_test[2])

#-------------------SDAE-LSTM--------------------
printer.print("-------------------SDAE-LSTM--------------------")

sdae_lstm_train = sdae.timeseries_encode(rnn_train[0])
sdae_lstm_test = sdae.timeseries_encode(rnn_test[0])
sdae_lstm = Lstm(scope_name='sdae-lstm', max_sequence_length=max_sequence_length, input_size=sdae_output_size, state_size=50, 
            output_size=output_size, loss_function='weighted-softmax-cross-entropy', initialization_function='xavier', metric_function='one-hot-accuracy',
            optimization_function='gradient-descent', learning_rate=0.05, learning_rate_decay='fraction', batch_size=32, 
            epochs=1, cost_mask=rnn_cost_mask, noise='none', printer=printer)

printer.print("Training SDAE-LSTM...")
sdae_lstm.train(sdae_lstm_train, rnn_train[1], rnn_train[2])
printer.print("Error on training set:")
sdae_lstm.test(sdae_lstm_train, rnn_train[1], rnn_train[2])
printer.print("Error on test set:")
sdae_lstm.test(sdae_lstm_test, rnn_test[1], rnn_test[2])
