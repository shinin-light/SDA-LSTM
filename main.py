import numpy as np
from networks import StackedAutoEncoder
from networks import ForwardClassifier
from networks import Lstm
from networks import Svm
from utils import Utils as utils
from sklearn import utils as skutils
from printer import Printer
import os

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
e_values_training = np.load("./data/e_records_training.npy")
e_classes_training = np.load("./data/e_classes_training.npy")
t_values_training = np.load("./data/t_records_training.npy")
t_classes_training = np.load("./data/t_classes_training.npy")

e_values_validation = np.load("./data/e_records_validation.npy")
e_classes_validation = np.load("./data/e_classes_validation.npy")
t_values_validation = np.load("./data/t_records_validation.npy")
t_classes_validation = np.load("./data/t_classes_validation.npy")

e_values_test = np.load("./data/e_records_test.npy")
e_classes_test = np.load("./data/e_classes_test.npy")
t_values_test = np.load("./data/t_records_test.npy")
t_classes_test = np.load("./data/t_classes_test.npy")

e_values_training = np.concatenate((e_values_training, e_classes_training), axis=2)
t_values_training = np.concatenate((t_values_training, t_classes_training), axis=2)

e_values_validation = np.concatenate((e_values_validation, e_classes_validation), axis=2)
t_values_validation = np.concatenate((t_values_validation, t_classes_validation), axis=2)

e_values_test = np.concatenate((e_values_test, e_classes_test), axis=2)
t_values_test = np.concatenate((t_values_test, t_classes_test), axis=2)

if not class_is_yesno and class_reduction is not None:
    e_classes_training = utils.reduce_classes_linear(e_classes_training, class_reduction)
    t_classes_training = utils.reduce_classes_linear(t_classes_training, class_reduction)

    e_classes_validation = utils.reduce_classes_linear(e_classes_validation, class_reduction)
    t_classes_validation = utils.reduce_classes_linear(t_classes_validation, class_reduction)

    e_classes_test = utils.reduce_classes_linear(e_classes_test, class_reduction)
    t_classes_test = utils.reduce_classes_linear(t_classes_test, class_reduction)

attributes_num = len(e_values_training[0][0])
classes_num = len(e_classes_training[0][0])

max_sequence_length = np.max([len(e_values_training[0]),len(t_values_training[0])])
e_values_training, e_classes_training, e_yesno_training, e_lengths_training = utils.rnn_shift_padding(e_values_training, e_classes_training, max_sequence_length)
t_values_training, t_classes_training, t_yesno_training, t_lengths_training = utils.rnn_shift_padding(t_values_training, t_classes_training, max_sequence_length)

e_values_validation, e_classes_validation, e_yesno_validation, e_lengths_validation = utils.rnn_shift_padding(e_values_validation, e_classes_validation, max_sequence_length)
t_values_validation, t_classes_validation, t_yesno_validation, t_lengths_validation = utils.rnn_shift_padding(t_values_validation, t_classes_validation, max_sequence_length)

e_values_test, e_classes_test, e_yesno_test, e_lengths_test = utils.rnn_shift_padding(e_values_test, e_classes_test, max_sequence_length)
t_values_test, t_classes_test, t_yesno_test, t_lengths_test = utils.rnn_shift_padding(t_values_test, t_classes_test, max_sequence_length)

#if(apply_reduction):
#    selection = np.random.choice(len(svm_e_values), min(len(svm_e_values), len(svmt_values)), replace=False)
#    svm_e_values, svm_e_classes = svm_e_values[selection], svm_e_classes[selection]

rnn_values_training = np.concatenate((e_values_training, t_values_training))
rnn_classes_training = np.concatenate((e_classes_training, t_classes_training))
rnn_yesno_training = np.concatenate((e_yesno_training, t_yesno_training))
rnn_lengths_training = np.concatenate((e_lengths_training, t_lengths_training))

rnn_values_validation = np.concatenate((e_values_validation, t_values_validation))
rnn_classes_validation = np.concatenate((e_classes_validation, t_classes_validation))
rnn_yesno_validation = np.concatenate((e_yesno_validation, t_yesno_validation))
rnn_lengths_validation = np.concatenate((e_lengths_validation, t_lengths_validation))

rnn_values_test = np.concatenate((e_values_test, t_values_test))
rnn_classes_test = np.concatenate((e_classes_test, t_classes_test))
rnn_yesno_test = np.concatenate((e_yesno_test, t_yesno_test))
rnn_lengths_test = np.concatenate((e_lengths_test, t_lengths_test))

flat_values_training = np.reshape(rnn_values_training,(-1, attributes_num))
flat_classes_training = np.reshape(rnn_classes_training,(-1, classes_num))
flat_yesno_training = np.reshape(rnn_yesno_training,(-1, 2))

flat_values_validation = np.reshape(rnn_values_validation,(-1, attributes_num))
flat_classes_validation = np.reshape(rnn_classes_validation,(-1, classes_num))
flat_yesno_validation = np.reshape(rnn_yesno_validation,(-1, 2))

flat_values_test = np.reshape(rnn_values_test,(-1, attributes_num))
flat_classes_test = np.reshape(rnn_classes_test,(-1, classes_num))
flat_yesno_test = np.reshape(rnn_yesno_test,(-1, 2))

if class_is_yesno :
    rnn_labels_training = rnn_yesno_training
    flat_labels_training = flat_yesno_training

    rnn_labels_validation = rnn_yesno_validation
    flat_labels_validation = flat_yesno_validation

    rnn_labels_test = rnn_yesno_test
    flat_labels_test = flat_yesno_test
else:
    rnn_labels_training = rnn_classes_training
    flat_labels_training = flat_classes_training

    rnn_labels_validation = rnn_classes_validation
    flat_labels_validation = flat_classes_validation

    rnn_labels_test = rnn_classes_test
    flat_labels_test = flat_classes_test

flat_values_training, flat_labels_training = utils.homogenize(flat_values_training, flat_labels_training, 0) #just remove the invalid records.
flat_values_validation, flat_labels_validation = utils.homogenize(flat_values_validation, flat_labels_validation, 0) #just remove the invalid records.
flat_values_test, flat_labels_test = utils.homogenize(flat_values_test, flat_labels_test, 0) #just remove the invalid records.

'''
sdae_values_training, sdae_labels_training = utils.homogenize(flat_values_training, flat_labels_training, 1) #balancing sdae training values.

sdae_train = np.concatenate((sdae_train_values, sdae_test_values))
sdae_finetuning_train = sdae_train_values
'''

rnn_cost_mask = utils.get_cost_mask(np.concatenate((rnn_labels_training, rnn_labels_validation)))
rnn_cost_mask /= np.mean(rnn_cost_mask)

flat_cost_mask = utils.get_cost_mask(np.concatenate((flat_labels_training, flat_labels_validation)))
flat_cost_mask /= np.mean(flat_cost_mask)

input_size = len(rnn_values_training[0][0])
output_size = len(rnn_labels_training[0][0])

print(input_size)
'''
weights = skutils.compute_class_weight(class_weight='balanced', classes=np.array(range(10)), y=np.argmax(flat_classes, 1))
alpha = 2


rnn_hidden_size = (len(rnn_train[0])) / (alpha * (input_size + output_size))
flat_hidden_size = (len(flat_train[0])) / (alpha * (input_size + output_size))
'''
#--------------------printer---------------------

printer = Printer()
'''
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

classifier = ForwardClassifier(scope_name='basic-forward', input_size=input_size, output_size=output_size, dims=[100], learning_rate_decay='exponential', 
                            activation_functions=['sigmoid'], loss_function='weighted-softmax-cross-entropy', cost_mask=flat_cost_mask,
                            metric_function='one-hot-accuracy', optimization_function='gradient-descent', epochs=100, learning_rate=0.05, batch_size=32, 
                            printer=printer, early_stop_lookahead=5)

printer.print("Training CLASSIFIER...")
classifier.train(flat_values_training, flat_labels_training, flat_values_validation, flat_labels_validation)
printer.print("Error on training set:")
classifier.test(flat_values_training, flat_labels_training)
printer.print("Error on validation set:")
classifier.test(flat_values_validation, flat_labels_validation)
printer.print("Error on test set:")
classifier.test(flat_values_test, flat_labels_test)

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
'''
#---------------------LSTM-----------------------
printer.print("---------------------LSTM-----------------------")

lstm = Lstm(scope_name='lstm', max_sequence_length=max_sequence_length, input_size=input_size, state_size=100, 
            output_size=output_size, loss_function='weighted-softmax-cross-entropy', initialization_function='xavier', metric_function='one-hot-accuracy',
            optimization_function='adam', learning_rate=0.001, learning_rate_decay='none', batch_size=32, 
            epochs=50, cost_mask=rnn_cost_mask, noise='none', printer=printer, early_stop_lookahead=5)

printer.print("Training LSTM...")
lstm.train(rnn_values_training, rnn_labels_training, rnn_lengths_training, rnn_values_validation, rnn_labels_validation, rnn_lengths_validation)
printer.print("Error on training set:")
lstm.test(rnn_values_training, rnn_labels_training, rnn_lengths_training)
printer.print("Error on validation set:")
lstm.test(rnn_values_validation, rnn_labels_validation, rnn_lengths_validation)
printer.print("Error on test set:")
lstm.test(rnn_values_test, rnn_labels_test, rnn_lengths_test)
'''
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
'''