import numpy as np
import tensorflow as tf
from networks import StackedAutoEncoder
from networks import ForwardClassifier
from networks import Lstm
from networks import Svm
from utils import Utils as utils
from sklearn import utils as skutils
from printer import Printer
import shutil
import os

training_frac = 0.8
apply_reduction = True
class_is_yesno = True
class_reduction = 5

#--------------------folders---------------------
shutil.rmtree('./logs', True) 

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

#--------------training-dictionary---------------

training_dictionary = {}
#training_dictionary['optimization_function'] = ['adam', 'gradient-descent']
#training_dictionary['activation_function'] = ['tanh', 'sigmoid', 'softmax', 'relu', 'softplus']
training_dictionary['one_hot_loss_function'] = ['weighted-softmax-rmse', 'weighted-softmax-cross-entropy']
training_dictionary['initialization_function'] = ['xavier', 'uniform', 'truncated-normal']
#training_dictionary['sdae_loss_function'] = ['rmse', 'sigmoid-cross-entropy']
#training_dictionary['epochs'] = [20]
training_dictionary['batch_size'] = [32, 64, 128]
training_dictionary['learning_rate'] = [[0.1, 'none'], [0.1, 'fraction'], [0.1, 'exponential'], [0.01, 'none'], [0.01, 'fraction'], [0.01, 'exponential'], [0.001, 'none'], [0.001, 'fraction'], [0.001, 'exponential']]
#training_dictionary['dims'] = [[200], [200, 200], [200, 200, 200], [100], [100, 50], [100, 50, 25], [50], [50, 25], [50, 25, 13]]
training_dictionary['state_size'] = [200, 100, 50]
'''
#------------------CLASSIFIER--------------------
print("------------------CLASSIFIER--------------------")

printer = Printer("results/classifier", True)
best_printer = Printer("results/classifier_best", False)
best_metrics = {}

counter = 0
for key in training_dictionary:
    
    for element in training_dictionary[key]:
        
        var = {}
        var['dims'] = element if key == 'dims' else [100]
        var['lr'] = element[0] if key == 'learning_rate' else 0.1
        var['lrd'] = element[1] if key == 'learning_rate' else 'fraction'
        var['af'] = [element if key == 'activation_function' else 'sigmoid' for i in range(len(var['dims']))]
        var['lf'] = element if key == 'one_hot_loss_function' else 'weighted-softmax-cross-entropy'
        var['of'] = element if key == 'optimization_function' else 'gradient-descent'
        var['epochs'] = element if key == 'epochs' else 80
        var['bs'] = element if key == 'batch_size' else 64
        var['if'] = element if key == 'initialization_function' else 'xavier'
       
        try:
            tf.reset_default_graph()
            
            scope_name='basic-forward-' + str(counter)
            counter += 1

            printer.open()
            printer.print('-----------------------------------------')
            printer.print("hyperparameters: {0}".format(var))

            classifier = ForwardClassifier(scope_name=scope_name, 
                                        input_size=input_size, 
                                        output_size=output_size, 
                                        dims=var['dims'], 
                                        learning_rate_decay=var['lrd'], 
                                        activation_functions=var['af'], 
                                        loss_function=var['lf'], 
                                        cost_mask=flat_cost_mask,
                                        metric_function='one-hot-accuracy', 
                                        optimization_function=var['of'], 
                                        epochs=var['epochs'], 
                                        learning_rate=var['lr'], 
                                        batch_size=var['bs'], 
                                        initialization_function=var['if'],
                                        printer=printer)
            
            printer.print("Training CLASSIFIER...")
            classifier.train(flat_train[0], flat_train[1])
            printer.print("Error on training set:")
            classifier.test(flat_train[0], flat_train[1])
            printer.print("Error on test set:")
            metrics = classifier.test(flat_test[0], flat_test[1])
            
            printer.close()

            found_best = False
            for m in metrics:
                key_value = (m, key)
                if key_value not in best_metrics:
                    best_metrics[key_value] = [metrics[m], element, var]
                    found_best = True
                elif (m == 'binary-brier-score' or m == 'emd'):
                    if metrics[m] < best_metrics[key_value][0]:
                        best_metrics[key_value] = [metrics[m], element, var]
                        found_best = True
                else:
                    if metrics[m] > best_metrics[key_value][0]:
                        best_metrics[key_value] = [metrics[m], element, var]
                        found_best = True
            
            if found_best:
                best_printer.open('w')
                for m in best_metrics:
                    best_printer.print("{0}: {1}".format(m, best_metrics[m]))
                best_printer.close()
        except BaseException as e:
            print(e)


counter = 0
for e1 in training_dictionary['dims']:
    
    for e2 in training_dictionary['learning_rate']:
        
        var = {}
        var['dims'] = e1
        var['lr'] = e2[0]
        var['lrd'] = e2[1]
        var['af'] = ['sigmoid' for i in range(len(var['dims']))]
        var['lf'] = 'weighted-softmax-cross-entropy'
        var['of'] = 'gradient-descent'
        var['epochs'] = 80
        var['bs'] = 64
        var['if'] = 'xavier'
       
        try:
            tf.reset_default_graph()
            scope_name='basic-forward-' + str(counter)
            counter += 1

            printer.open()
            printer.print('-----------------------------------------')
            printer.print("hyperparameters of {0}: {1}".format(scope_name,var))

            classifier = ForwardClassifier(scope_name=scope_name, 
                                        input_size=input_size, 
                                        output_size=output_size, 
                                        dims=var['dims'], 
                                        learning_rate_decay=var['lrd'], 
                                        activation_functions=var['af'], 
                                        loss_function=var['lf'], 
                                        cost_mask=flat_cost_mask,
                                        metric_function='one-hot-accuracy', 
                                        optimization_function=var['of'], 
                                        epochs=var['epochs'], 
                                        learning_rate=var['lr'], 
                                        batch_size=var['bs'], 
                                        initialization_function=var['if'],
                                        printer=printer)
            
            printer.print("Training CLASSIFIER...")
            classifier.train(flat_train[0], flat_train[1])
            printer.print("Error on training set:")
            classifier.test(flat_train[0], flat_train[1])
            printer.print("Error on test set:")
            metrics = classifier.test(flat_test[0], flat_test[1])
            
            del classifier
            printer.close()

            found_best = False
            for m in metrics:
                if m not in best_metrics:
                    best_metrics[m] = [metrics[m], e1, e2, var]
                    found_best = True
                elif (m == 'binary-brier-score' or m == 'emd'):
                    if metrics[m] < best_metrics[m][0]:
                        best_metrics[m] = [metrics[m], e1, e2, var]
                        found_best = True
                else:
                    if metrics[m] > best_metrics[m][0]:
                        best_metrics[m] = [metrics[m], e1, e2, var]
                        found_best = True
            
            if found_best:
                best_printer.open('w')
                for m in best_metrics:
                    best_printer.print("{0}: {1}".format(m, best_metrics[m]))
                best_printer.close()
        except BaseException as e:
            print(e)

'''
#---------------------LSTM-----------------------
print("---------------------LSTM-----------------------")

printer = Printer("results/lstm_norm", True)
best_printer = Printer("results/lstm_norm_best", False)
best_metrics = {}

counter = 0
for key in training_dictionary:
    
    for element in training_dictionary[key]:
        
        var = {}
        var['ss'] = element if key == 'state_size' else 100
        var['lr'] = element[0] if key == 'learning_rate' else 0.1
        var['lrd'] = element[1] if key == 'learning_rate' else 'exponential'
        var['lf'] = element if key == 'one_hot_loss_function' else 'weighted-softmax-cross-entropy'
        var['of'] = element if key == 'optimization_function' else 'gradient-descent'
        var['epochs'] = element if key == 'epochs' else 20
        var['bs'] = element if key == 'batch_size' else 32
        var['if'] = element if key == 'initialization_function' else 'xavier'
       
        try:
            tf.reset_default_graph()
            
            scope_name='basic-lstm-' + str(counter)
            counter += 1

            printer.open()
            printer.print('-----------------------------------------')
            printer.print("hyperparameters: {0}".format(var))

            lstm = Lstm(scope_name=scope_name, 
                        max_sequence_length=max_sequence_length, 
                        input_size=input_size, 
                        state_size=var['ss'], 
                        output_size=output_size, 
                        loss_function=var['lf'], 
                        initialization_function=var['if'], 
                        metric_function='one-hot-accuracy',
                        optimization_function=var['of'], 
                        learning_rate=var['lr'], 
                        learning_rate_decay=var['lrd'], 
                        batch_size=var['bs'], 
                        epochs=var['epochs'],
                        cost_mask=rnn_cost_mask, 
                        noise='none', 
                        printer=printer)
            
            printer.print("Training LSTM...")
            lstm.train(rnn_train[0], rnn_train[1], rnn_train[2])
            printer.print("Error on training set:")
            lstm.test(rnn_train[0], rnn_train[1], rnn_train[2])
            printer.print("Error on test set:")
            metrics = lstm.test(rnn_test[0], rnn_test[1], rnn_test[2])
            
            printer.close()

            found_best = False
            for m in metrics:
                key_value = (m, key)
                if key_value not in best_metrics:
                    best_metrics[key_value] = [metrics[m], element, var]
                    found_best = True
                elif (m == 'binary-brier-score' or m == 'emd'):
                    if metrics[m] < best_metrics[key_value][0]:
                        best_metrics[key_value] = [metrics[m], element, var]
                        found_best = True
                else:
                    if metrics[m] > best_metrics[key_value][0]:
                        best_metrics[key_value] = [metrics[m], element, var]
                        found_best = True
            
            if found_best:
                best_printer.open('w')
                for m in best_metrics:
                    best_printer.print("{0}: {1}".format(m, best_metrics[m]))
                best_printer.close()
        except BaseException as e:
            print(e)