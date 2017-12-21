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
import datetime

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

#SDAE
flat_e_values_training = np.reshape(e_values_training,(-1, attributes_num))
flat_e_labels_training = np.reshape(e_yesno_training,(-1, 2))
flat_e_values_training, flat_e_labels_training = utils.homogenize(flat_e_values_training, flat_e_labels_training, 0) #just remove the invalid records.
sdae_e_values_training, _ = utils.homogenize(flat_e_values_training, flat_e_labels_training, 1) #SDAE oversampling

flat_e_values_validation = np.reshape(e_values_validation,(-1, attributes_num))
flat_e_labels_validation = np.reshape(e_yesno_validation,(-1, 2))
flat_e_values_validation, flat_e_labels_validation = utils.homogenize(flat_e_values_validation, flat_e_labels_validation, 0) #just remove the invalid records.

flat_e_values_test = np.reshape(e_values_test,(-1, attributes_num))
flat_e_labels_test = np.reshape(e_yesno_test,(-1, 2))
flat_e_values_test, flat_e_labels_test = utils.homogenize(flat_e_values_test, flat_e_labels_test, 0) #just remove the invalid records.

flat_t_values_training = np.reshape(t_values_training,(-1, attributes_num))
flat_t_labels_training = np.reshape(t_yesno_training,(-1, 2))
flat_t_values_training, flat_t_labels_training = utils.homogenize(flat_t_values_training, flat_t_labels_training, 0) #just remove the invalid records.
sdae_t_values_training, _ = utils.homogenize(flat_t_values_training, flat_t_labels_training, 1) #SDAE oversampling

flat_t_values_validation = np.reshape(t_values_validation,(-1, attributes_num))
flat_t_labels_validation = np.reshape(t_yesno_validation,(-1, 2))
flat_t_values_validation, flat_t_labels_validation = utils.homogenize(flat_t_values_validation, flat_t_labels_validation, 0) #just remove the invalid records.

flat_t_values_test = np.reshape(t_values_test,(-1, attributes_num))
flat_t_labels_test = np.reshape(t_yesno_test,(-1, 2))
flat_t_values_test, flat_t_labels_test = utils.homogenize(flat_t_values_test, flat_t_labels_test, 0) #just remove the invalid records.

sdae_layerwise_training = np.concatenate((sdae_e_values_training, sdae_t_values_training))
sdae_layerwise_validation = np.concatenate((flat_e_values_validation, flat_t_values_validation))
sdae_finetuning_training = [sdae_e_values_training, sdae_t_values_training]

sdae_flat_values_training = [flat_e_values_training, flat_t_values_training]
sdae_flat_labels_training = [flat_e_labels_training, flat_t_labels_training]

sdae_flat_values_validation = [flat_e_values_validation, flat_t_values_validation]
sdae_flat_labels_validation = [flat_e_labels_validation, flat_t_labels_validation]

sdae_flat_values_test = [flat_e_values_test, flat_t_values_test]
sdae_flat_labels_test = [flat_e_labels_test, flat_t_labels_test]

sdae_rnn_values_training = [e_values_training, t_values_training]
sdae_rnn_labels_training = [e_yesno_training, t_yesno_training]
sdae_rnn_lengths_training = [e_lengths_training, t_lengths_training]

sdae_rnn_values_validation = [e_values_validation, t_values_validation]
sdae_rnn_labels_validation = [e_yesno_validation, t_yesno_validation]
sdae_rnn_lengths_validation = [e_lengths_validation, t_lengths_validation]

sdae_rnn_values_test = [e_values_test, t_values_test]
sdae_rnn_labels_test = [e_yesno_test, t_yesno_test]
sdae_rnn_lengths_test = [e_lengths_test, t_lengths_test]

#END-SDAE

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

'''
weights = skutils.compute_class_weight(class_weight='balanced', classes=np.array(range(10)), y=np.argmax(flat_classes, 1))
alpha = 2


rnn_hidden_size = (len(rnn_train[0])) / (alpha * (input_size + output_size))
flat_hidden_size = (len(flat_train[0])) / (alpha * (input_size + output_size))
'''
#--------------training-dictionary---------------

training_dictionary = {}
#training_dictionary['one_hot_loss_function'] = ['weighted-softmax-rmse', 'weighted-softmax-cross-entropy']
#training_dictionary['initialization_function'] = ['xavier', 'uniform', 'truncated-normal']
#training_dictionary['sdae_loss_function'] = ['rmse', 'sigmoid-cross-entropy']
#training_dictionary['epochs'] = [20]
#training_dictionary['learning_rate'] = [[0.01, 'exponential'], [0.001, 'exponential']]
#training_dictionary['activation_function'] = ['tanh', 'sigmoid', 'softplus', 'relu']
#training_dictionary['batch_size'] = [64, 128, 256]
#training_dictionary['dims'] = [[300, 300], [150, 150], [150, 150, 150], [300, 300, 300], [200, 100], [100, 50], [200, 100, 50], [300, 200, 100]]
#training_dictionary['optimization_function'] = ['adam']
#training_dictionary['noise'] = ['gaussian']
#training_dictionary['state_size'] = [50, 100, 200] # [[100],[200],[300],[150,150],[200,100],[100,50],[150,150,150],[200,100,50],[300,200,100]]

'''
#------------------CLASSIFIER--------------------
print("------------------CLASSIFIER--------------------")

printer = Printer("results/classifier_deep_noise", True)
best_printer = Printer("results/classifier_deep_noise_best", False)
best_metrics = {}

counter = 0
for lr in training_dictionary['learning_rate']:
    for af in training_dictionary['activation_function']:
        for bs in training_dictionary['batch_size']:
            for ss in training_dictionary['dims']:
                try:
                    tf.reset_default_graph()
                    
                    af2 = [af]
                    for i in range(len(ss) - 1):
                        af2.append(af)

                    scope_name='classifier-' + str(counter)
                    counter += 1

                    printer.open()
                    printer.print('-----------------------------------------')
                    printer.print("hyperparameters: {0}".format([lr[0], lr[1], af, bs, ss]))

                    classifier = ForwardClassifier(scope_name=scope_name, 
                                                input_size=input_size, 
                                                output_size=output_size, 
                                                dims=ss, 
                                                learning_rate_decay=lr[1], 
                                                activation_functions=af2, 
                                                loss_function="weighted-softmax-cross-entropy", 
                                                cost_mask=flat_cost_mask,
                                                metric_function='one-hot-accuracy', 
                                                optimization_function='adam', 
                                                epochs=100, 
                                                learning_rate=lr[0], 
                                                batch_size=bs, 
                                                initialization_function='xavier',
                                                early_stop_lookahead=10,
                                                noise='gaussian',
                                                printer=printer)
                    
                    printer.print("Training CLASSIFIER...")
                    classifier.train(flat_values_training, flat_labels_training, flat_values_validation, flat_labels_validation)
                    printer.print("Error on training set:")
                    classifier.test(flat_values_training, flat_labels_training)
                    printer.print("Error on validation set:")
                    metrics = classifier.test(flat_values_validation, flat_labels_validation)
                    printer.print("Error on test set:")
                    classifier.test(flat_values_test, flat_labels_test)
                    
                    del classifier
                    printer.close()

                    found_best = False
                    for m in metrics:
                        if m not in best_metrics:
                            best_metrics[m] = [metrics[m], lr[0], lr[1], af, bs, ss]
                            found_best = True
                        elif (m == 'binary-brier-score' or m == 'emd'):
                            if metrics[m] < best_metrics[m][0]:
                                best_metrics[m] = [metrics[m], lr[0], lr[1], af, bs, ss]
                                found_best = True
                        else:
                            if metrics[m] > best_metrics[m][0]:
                                best_metrics[m] = [metrics[m], lr[0], lr[1], af, bs, ss]
                                found_best = True
                    
                    if found_best:
                        best_printer.open('w')
                        for m in best_metrics:
                            best_printer.print("{0}: {1}".format(m, best_metrics[m]))
                        best_printer.close()

                except BaseException as e:
                    print(e)
        


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


#---------------------LSTM-----------------------
print("---------------------LSTM-----------------------")

printer = Printer("results/lstm_noise", True)
best_printer = Printer("results/lstm_noise_best", False)
best_metrics = {}

counter = 0
for of in training_dictionary['optimization_function']:
    for lr in training_dictionary['learning_rate']:
        for noise in training_dictionary['noise']:
            for ss in training_dictionary['state_size']:
                for bs in training_dictionary['batch_size']:
                    try:
                        tf.reset_default_graph()
                        
                        scope_name='basic-lstm-' + str(counter)
                        counter += 1

                        printer.open()
                        printer.print('-----------------------------------------')
                        printer.print("hyperparameters: {0}".format([of, lr[0], lr[1], noise, ss, bs]))

                        lstm = Lstm(scope_name=scope_name, 
                                    max_sequence_length=max_sequence_length, 
                                    input_size=input_size, 
                                    state_size=ss, 
                                    output_size=output_size, 
                                    loss_function="weighted-softmax-cross-entropy", 
                                    initialization_function="xavier", 
                                    metric_function='one-hot-accuracy',
                                    optimization_function=of, 
                                    learning_rate=lr[0], 
                                    learning_rate_decay=lr[1], 
                                    batch_size=bs, 
                                    epochs=100,
                                    cost_mask=rnn_cost_mask, 
                                    noise=noise, 
                                    early_stop_lookahead=10,
                                    printer=printer)
                        
                        printer.print("Training LSTM...")
                        lstm.train(rnn_values_training, rnn_labels_training, rnn_lengths_training, rnn_values_validation, rnn_labels_validation, rnn_lengths_validation)
                        printer.print("Error on training set:")
                        lstm.test(rnn_values_training, rnn_labels_training, rnn_lengths_training)
                        printer.print("Error on validation set:")
                        metrics = lstm.test(rnn_values_validation, rnn_labels_validation, rnn_lengths_validation)
                        printer.print("Error on test set:")
                        lstm.test(rnn_values_test, rnn_labels_test, rnn_lengths_test)
                        
                        del lstm
                        printer.close()

                        found_best = False
                        for m in metrics:
                            if m not in best_metrics:
                                best_metrics[m] = [metrics[m], of, lr[0], lr[1], noise, ss, bs]
                                found_best = True
                            elif (m == 'binary-brier-score' or m == 'emd'):
                                if metrics[m] < best_metrics[m][0]:
                                    best_metrics[m] = [metrics[m], of, lr[0], lr[1], noise, ss, bs]
                                    found_best = True
                            else:
                                if metrics[m] > best_metrics[m][0]:
                                    best_metrics[m] = [metrics[m], of, lr[0], lr[1], noise, ss, bs]
                                    found_best = True
                        
                        if found_best:
                            best_printer.open('w')
                            for m in best_metrics:
                                best_printer.print("{0}: {1}".format(m, best_metrics[m]))
                            best_printer.close()

                    except BaseException as e:
                        print(e)


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
'''


#---------------------SDAE-----------------------
print("---------------------SDAE-----------------------")


training_dictionary['activation_function'] = ['tanh', 'relu']
#training_dictionary['initialization_function'] = ['xavier', 'uniform', 'truncated-normal']
training_dictionary['sdae_loss_function'] = ['rmse', 'sigmoid-cross-entropy']
#training_dictionary['epochs'] = [20]
training_dictionary['batch_size'] = [128]
training_dictionary['dims'] = [[300,300,300], [300,150,50], [150,75,30], [500], [300], [100]]
#training_dictionary['optimization_function'] = ['adam']
training_dictionary['learning_rate'] = [[0.01, 'exponential'], [0.001, 'exponential']]
training_dictionary['noise'] = ['mask-0.3', 'mask-0.6', 'mask-0.9']
#training_dictionary['state_size'] = [50, 100, 200] # [[100],[200],[300],[150,150],[200,100],[100,50],[150,150,150],[200,100,50],[300,200,100]]


number_of_hyperparameters = 1
for h in training_dictionary:
    number_of_hyperparameters *= len(training_dictionary[h])

print(number_of_hyperparameters)

printer = Printer("results/sdae_BLA", True)
printer.open()
best_printer = Printer("results/sdae_best_BLA", False)
best_metrics = [{},{},{},{}]
pre_metrics = [{},{},{},{}]
dataset = 0


#bad stuff, fix later…
sdae_rnn_values_training[dataset] = np.concatenate((sdae_rnn_values_training[dataset], sdae_rnn_values_training[1-dataset]))
sdae_rnn_labels_training[dataset] = np.concatenate((sdae_rnn_labels_training[dataset], sdae_rnn_labels_training[1-dataset]))
sdae_rnn_lengths_training[dataset] = np.concatenate((sdae_rnn_lengths_training[dataset], sdae_rnn_lengths_training[1-dataset]))

sdae_rnn_values_validation[dataset] = np.concatenate((sdae_rnn_values_validation[dataset], sdae_rnn_values_validation[1-dataset]))
sdae_rnn_labels_validation[dataset] = np.concatenate((sdae_rnn_labels_validation[dataset], sdae_rnn_labels_validation[1-dataset]))
sdae_rnn_lengths_validation[dataset] = np.concatenate((sdae_rnn_lengths_validation[dataset], sdae_rnn_lengths_validation[1-dataset]))

sdae_rnn_values_test[dataset] = np.concatenate((sdae_rnn_values_test[dataset], sdae_rnn_values_test[1-dataset]))
sdae_rnn_labels_test[dataset] = np.concatenate((sdae_rnn_labels_test[dataset], sdae_rnn_labels_test[1-dataset]))
sdae_rnn_lengths_test[dataset] = np.concatenate((sdae_rnn_lengths_test[dataset], sdae_rnn_lengths_test[1-dataset]))


sdae_flat_values_training[dataset] = np.concatenate((sdae_flat_values_training[dataset], sdae_flat_values_training[1-dataset]))
sdae_flat_labels_training[dataset] = np.concatenate((sdae_flat_labels_training[dataset], sdae_flat_labels_training[1-dataset]))

sdae_flat_values_validation[dataset] = np.concatenate((sdae_flat_values_validation[dataset], sdae_flat_values_validation[1-dataset]))
sdae_flat_labels_validation[dataset] = np.concatenate((sdae_flat_labels_validation[dataset], sdae_flat_labels_validation[1-dataset]))

sdae_flat_values_test[dataset] = np.concatenate((sdae_flat_values_test[dataset], sdae_flat_values_test[1-dataset]))
sdae_flat_labels_test[dataset] = np.concatenate((sdae_flat_labels_test[dataset], sdae_flat_labels_test[1-dataset]))
#end bad stuff

pre_forward_scope_name='pre_forward'
pre_forward_deep_scope_name='pre_forward-deep'
pre_lstm_scope_name='pre_lstm'
'''
#------------LSTM-----------
pre_lstm = Lstm(scope_name=pre_lstm_scope_name, 
            max_sequence_length=max_sequence_length, 
            input_size=input_size, 
            state_size=200, 
            output_size=output_size, 
            loss_function="weighted-softmax-cross-entropy", 
            initialization_function="xavier", 
            metric_function='one-hot-accuracy',
            optimization_function='adam', 
            learning_rate=0.001, 
            learning_rate_decay='exponential', 
            batch_size=128,
            epochs=100,
            cost_mask=rnn_cost_mask, 
            noise='gaussian', 
            early_stop_lookahead=5,
            printer=printer)

#TODO: add encoded
printer.print("Training PRE LSTM...")
pre_lstm.train(sdae_rnn_values_training[dataset], sdae_rnn_labels_training[dataset], sdae_rnn_lengths_training[dataset], sdae_rnn_values_validation[dataset], sdae_rnn_labels_validation[dataset], sdae_rnn_lengths_validation[dataset])
printer.print("Error on training set:")
pre_lstm.test(sdae_rnn_values_training[dataset], sdae_rnn_labels_training[dataset], sdae_rnn_lengths_training[dataset])
printer.print("Error on validation set:")
pre_metrics[0] = pre_lstm.test(sdae_rnn_values_validation[dataset], sdae_rnn_labels_validation[dataset], sdae_rnn_lengths_validation[dataset])
printer.print("Error on test set:")
pre_lstm.test(sdae_rnn_values_test[dataset], sdae_rnn_labels_test[dataset], sdae_rnn_lengths_test[dataset])

#----------forward----------
pre_deep_classifier = ForwardClassifier(scope_name=pre_forward_deep_scope_name, 
                        input_size=input_size, 
                        output_size=output_size, 
                        dims=[300,300,300], 
                        learning_rate_decay='exponential', 
                        activation_functions=['tanh', 'tanh', 'tanh'], 
                        loss_function="weighted-softmax-cross-entropy", 
                        cost_mask=flat_cost_mask,
                        metric_function='one-hot-accuracy', 
                        optimization_function='adam', 
                        epochs=100, 
                        learning_rate=0.001, 
                        batch_size=128,
                        noise='gaussian',
                        initialization_function='xavier',
                        early_stop_lookahead=5,
                        printer=printer)

#TODO: add encoded
printer.print("Training PRE DEEP CLASSIFIER...")
pre_deep_classifier.train(sdae_flat_values_training[dataset], sdae_flat_labels_training[dataset], sdae_flat_values_validation[dataset], sdae_flat_labels_validation[dataset])
printer.print("Error on training set:")
pre_deep_classifier.test(sdae_flat_values_training[dataset], sdae_flat_labels_training[dataset])
printer.print("Error on validation set:")
pre_metrics[1] = pre_deep_classifier.test(sdae_flat_values_validation[dataset], sdae_flat_labels_validation[dataset])
printer.print("Error on test set:")
pre_deep_classifier.test(sdae_flat_values_test[dataset], sdae_flat_labels_test[dataset])
'''                      
#----------forward----------
pre_classifier = ForwardClassifier(scope_name=pre_forward_scope_name, 
                        input_size=input_size, 
                        output_size=output_size, 
                        dims=[400], 
                        learning_rate_decay='exponential', 
                        activation_functions=['tanh'], 
                        loss_function="weighted-softmax-cross-entropy", 
                        cost_mask=flat_cost_mask,
                        metric_function='one-hot-accuracy', 
                        optimization_function='adam', 
                        epochs=1, 
                        learning_rate=0.01, 
                        batch_size=128,
                        noise='gaussian',
                        initialization_function='xavier',
                        early_stop_lookahead=1,
                        printer=printer)

#TODO: add encoded
printer.print("Training PRE CLASSIFIER...")
pre_classifier.train(sdae_flat_values_training[dataset], sdae_flat_labels_training[dataset], sdae_flat_values_validation[dataset], sdae_flat_labels_validation[dataset])
printer.print("Error on training set:")
pre_classifier.test(sdae_flat_values_training[dataset], sdae_flat_labels_training[dataset])
printer.print("Error on validation set:")
pre_metrics[2] = pre_classifier.test(sdae_flat_values_validation[dataset], sdae_flat_labels_validation[dataset])
printer.print("Error on test set:")
pre_classifier.test(sdae_flat_values_test[dataset], sdae_flat_labels_test[dataset])

#-----------SVM-----------
pre_svm = Svm(cost_mask=flat_cost_mask, output_size=output_size, printer=printer, batch_size=256)
printer.print("Training PRE SVM... ")
pre_svm.train(sdae_flat_values_training[dataset], sdae_flat_labels_training[dataset])
printer.print("Error on training set:")
pre_svm.test(sdae_flat_values_training[dataset], sdae_flat_labels_training[dataset])
printer.print("Error on validation set:")
pre_metrics[3] = pre_svm.test(sdae_flat_values_validation[dataset], sdae_flat_labels_validation[dataset])
printer.print("Error on test set:")
pre_svm.test(sdae_flat_values_test[dataset], sdae_flat_labels_test[dataset])

printer.close()
del pre_lstm
del pre_deep_classifier
del pre_classifier
del pre_svm

counter = 0
for af in training_dictionary['activation_function']:
    for lf in training_dictionary['sdae_loss_function']:
        for lr in training_dictionary['learning_rate']:
            for noise in training_dictionary['noise']:
                for ss in training_dictionary['dims']:
                    for bs in training_dictionary['batch_size']:
                        try:
                            tf.reset_default_graph()
                            post_metrics = [{},{},{},{}]

                            sdae_scope_name='sdae-' + str(counter)
                            post_lstm_scope_name='post_lstm-' + str(counter)
                            post_forward_scope_name='post_forward-' + str(counter)
                            post_forward_deep_scope_name='post_forward-deep-' + str(counter)
                            counter += 1

                            printer.open()
                            printer.print('-----------------------------------------')
                            printer.print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
                            printer.print('Hyperparameter {0} of {1}'.format(counter, number_of_hyperparameters))
                            printer.print("hyperparameters: {0}".format([af, lf, lr[0], lr[1], noise, ss, bs]))

                            sdae_output_size = ss[-1]
                            
                            ef = [af]
                            for i in range(len(ss) - 1):
                                ef.append(af)

                            lf2 = [lf]
                            for i in range(len(ss) - 1):
                                lf2.append('rmse')

                            df = ['sigmoid'] if lf == 'rmse' else ['linear']
                            for i in range(len(ss) - 1):
                                tmp = af if lf == 'rmse' else 'linear'
                                df.append(tmp)
                            
                            noise2 = [noise]
                            for i in range(len(ss) - 1):
                                noise2.append('gaussian')

                            sdae = StackedAutoEncoder(scope_name=sdae_scope_name, 
                                input_size=input_size, 
                                dims=ss, 
                                optimization_function='adam',
                                encoding_functions=ef, 
                                decoding_functions=df,
                                noise=noise2, 
                                epochs=100, 
                                loss_functions=lf2, 
                                learning_rate=lr[0], 
                                learning_rate_decay=lr[1], 
                                batch_size=bs, 
                                early_stop_lookahead=5,
                                printer=printer)

                            printer.print("Training SDAE...")
                            sdae.train(sdae_layerwise_training, sdae_layerwise_validation)
                            printer.print("Finetuning SDAE...")
                            sdae.finetune(sdae_finetuning_training[dataset], sdae_flat_values_validation[dataset])

                            encoded_flat_training = sdae.encode(sdae_flat_values_training[dataset])
                            encoded_flat_validation = sdae.encode(sdae_flat_values_validation[dataset])
                            encoded_flat_test = sdae.encode(sdae_flat_values_test[dataset])

                            encoded_rnn_training = sdae.timeseries_encode(sdae_rnn_values_training[dataset])
                            encoded_rnn_validation = sdae.timeseries_encode(sdae_rnn_values_validation[dataset])
                            encoded_rnn_test = sdae.timeseries_encode(sdae_rnn_values_test[dataset])

                            #------------LSTM-----------
                            post_lstm = Lstm(scope_name=post_lstm_scope_name, 
                                        max_sequence_length=max_sequence_length, 
                                        input_size=sdae_output_size, 
                                        state_size=200, 
                                        output_size=output_size, 
                                        loss_function="weighted-softmax-cross-entropy", 
                                        initialization_function="xavier", 
                                        metric_function='one-hot-accuracy',
                                        optimization_function='adam', 
                                        learning_rate=0.001, 
                                        learning_rate_decay='exponential', 
                                        batch_size=128,
                                        epochs=100,
                                        cost_mask=rnn_cost_mask, 
                                        noise='gaussian', 
                                        early_stop_lookahead=5,
                                        printer=printer)

                            #TODO: add encoded
                            printer.print("Training POST LSTM...")
                            post_lstm.train(encoded_rnn_training, sdae_rnn_labels_training[dataset], sdae_rnn_lengths_training[dataset], encoded_rnn_validation, sdae_rnn_labels_validation[dataset], sdae_rnn_lengths_validation[dataset])
                            printer.print("Error on training set:")
                            post_lstm.test(encoded_rnn_training, sdae_rnn_labels_training[dataset], sdae_rnn_lengths_training[dataset])
                            printer.print("Error on validation set:")
                            post_metrics[0] = post_lstm.test(encoded_rnn_validation, sdae_rnn_labels_validation[dataset], sdae_rnn_lengths_validation[dataset])
                            printer.print("Error on test set:")
                            post_lstm.test(encoded_rnn_test, sdae_rnn_labels_test[dataset], sdae_rnn_lengths_test[dataset])

                            
                            #----------forward----------
                            post_deep_classifier = ForwardClassifier(scope_name=post_forward_deep_scope_name, 
                                                    input_size=sdae_output_size, 
                                                    output_size=output_size, 
                                                    dims=[300,300,300], 
                                                    learning_rate_decay='exponential', 
                                                    activation_functions=['tanh', 'tanh', 'tanh'], 
                                                    loss_function="weighted-softmax-cross-entropy", 
                                                    cost_mask=flat_cost_mask,
                                                    metric_function='one-hot-accuracy', 
                                                    optimization_function='adam', 
                                                    epochs=100, 
                                                    learning_rate=0.001, 
                                                    batch_size=128,
                                                    noise='gaussian',
                                                    initialization_function='xavier',
                                                    early_stop_lookahead=5,
                                                    printer=printer)

                            #TODO: add encoded
                            printer.print("Training POST DEEP CLASSIFIER...")
                            post_deep_classifier.train(encoded_flat_training, sdae_flat_labels_training[dataset], encoded_flat_validation, sdae_flat_labels_validation[dataset])
                            printer.print("Error on training set:")
                            post_deep_classifier.test(encoded_flat_training, sdae_flat_labels_training[dataset])
                            printer.print("Error on validation set:")
                            post_metrics[1] = post_deep_classifier.test(encoded_flat_validation, sdae_flat_labels_validation[dataset])
                            printer.print("Error on test set:")
                            post_deep_classifier.test(encoded_flat_test, sdae_flat_labels_test[dataset])
                                                        
                            #----------forward----------
                            post_classifier = ForwardClassifier(scope_name=post_forward_scope_name, 
                                                    input_size=sdae_output_size, 
                                                    output_size=output_size, 
                                                    dims=[400], 
                                                    learning_rate_decay='exponential', 
                                                    activation_functions=['tanh'], 
                                                    loss_function="weighted-softmax-cross-entropy", 
                                                    cost_mask=flat_cost_mask,
                                                    metric_function='one-hot-accuracy', 
                                                    optimization_function='adam', 
                                                    epochs=100, 
                                                    learning_rate=0.01, 
                                                    batch_size=128,
                                                    noise='gaussian',
                                                    initialization_function='xavier',
                                                    early_stop_lookahead=5,
                                                    printer=printer)

                            #TODO: add encoded
                            printer.print("Training POST CLASSIFIER...")
                            post_classifier.train(encoded_flat_training, sdae_flat_labels_training[dataset], encoded_flat_validation, sdae_flat_labels_validation[dataset])
                            printer.print("Error on training set:")
                            post_classifier.test(encoded_flat_training, sdae_flat_labels_training[dataset])
                            printer.print("Error on validation set:")
                            post_metrics[2] = post_classifier.test(encoded_flat_validation, sdae_flat_labels_validation[dataset])
                            printer.print("Error on test set:")
                            post_classifier.test(encoded_flat_test, sdae_flat_labels_test[dataset])

                            #-----------SVM-----------
                            post_svm = Svm(cost_mask=flat_cost_mask, output_size=output_size, printer=printer, batch_size=256)
                            printer.print("Training POST SVM... ")
                            post_svm.train(encoded_flat_training, sdae_flat_labels_training[dataset])
                            printer.print("Error on training set:")
                            post_svm.test(encoded_flat_training, sdae_flat_labels_training[dataset])
                            printer.print("Error on validation set:")
                            post_metrics[3] = post_svm.test(encoded_flat_validation, sdae_flat_labels_validation[dataset])
                            printer.print("Error on test set:")
                            post_svm.test(encoded_flat_test, sdae_flat_labels_test[dataset])

                            #----------METRIC---------
                            del sdae
                            del post_lstm
                            del post_deep_classifier
                            del post_classifier
                            del post_svm
                            printer.close()

                            #TODO: LSTM_METRICS, CLASSIFIER_DEEP_METRICS, CLASSIFIER_METRICS, SVM_METRICS
                            found_best = False
                            for i in range(len(post_metrics)):
                                for m in post_metrics[i]:
                                    increment = post_metrics[i][m] - pre_metrics[i][m]
                                    if m not in best_metrics[i]:
                                        best_metrics[i][m] = [increment, post_metrics[i][m], pre_metrics[i][m], af, lf, lr[0], lr[1], noise, ss, bs]
                                        found_best = True
                                    elif (m == 'binary-brier-score' or m == 'emd'):
                                        if increment < best_metrics[i][m][0]:
                                            best_metrics[i][m] = [increment, post_metrics[i][m], pre_metrics[i][m], af, lf, lr[0], lr[1], noise, ss, bs]
                                            found_best = True
                                    else:
                                        if increment > best_metrics[i][m][0]:
                                            best_metrics[i][m] = [increment, post_metrics[i][m], pre_metrics[i][m], af, lf, lr[0], lr[1], noise, ss, bs]
                                            found_best = True
                            
                            if found_best:
                                best_printer.open('w')
                                for i in range(len(post_metrics)):
                                    for m in post_metrics[i]:
                                        best_printer.print("Classifier {0}, metric {1}: {2}".format(i, m, best_metrics[i][m]))                          
                                best_printer.close()

                        except BaseException as e:
                            print(e)