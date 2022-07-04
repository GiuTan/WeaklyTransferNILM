import os
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import gc
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import precision_recall_curve, classification_report, auc
from CRNN_t import *
from utils_func import *
import json
import random as python_random
import tensorflow as tf
from metrics import *
from params import params, uk_params, refit_params

if __name__ == '__main__':
    # UK-DALE path
    path = '../weak_labels/'
    file_agg_path =  path + 'dataset_weak/aggregate_data_noised/'
    file_labels_path = path + 'dataset_weak/labels/'
    model_ = 'strong_weakUK'

    # REFIT path
    WEAK_agg_resample_path = '../resampled_agg_REFIT/'

    # set seeds for reproducible results
    random.seed(123)
    np.random.seed(123)
    python_random.seed(123)
    tf.random.set_seed(1234)
    tf.experimental.numpy.random.seed(1234)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    os.environ["CUDA_VISIBLE_DEVICES"]="6"

    # UKDALE DATA QUANTITIES

    quantity_1_u = 45581
    quantity_2_u = 3271
    quantity_5_u = 2969
    quantity_4_u = 553
    quantity_3_u = 3047

    # REFIT DATA QUANTITIES

    quantity_2_r = 3000
    quantity_4_r = 12000
    quantity_8_r = 11000
    quantity_9_r = 9000
    quantity_15_r = 1500


    # Flag Inizialization
    test = True

    strong = False
    strong_weak = True #se avessi voluto escludere da alcuni segmenti le labels weak
    test_ukdale = False
    weak_counter = True
    val_only_weak_uk = False
    validation_refit = False
    val_only_weak_re = False

    X_train, Y_train, Y_train_weak = np.load([]),np.load([]),np.load([]) # REFIT fine-tuning set  
    X_test, Y_test, Y_test_weak = np.load([]),np.load([]),np.load([]) # REFIT test set
    X_val, Y_val, Y_val_weak = np.load([]),np.load([]),np.load([])   # UKDALE validation set
    # number of weak labels considered 
    weak_count(Y_train_weak)

    print("Total x train",X_train.shape)
    print("Total Y train", Y_train.shape)
    print("Total Y train weak", Y_train_weak.shape)
    assert (len(Y_val) == len(Y_val_weak))
    assert (len(Y_train) == len(Y_train_weak))

    x_train = X_train
    y_strong_train = Y_train
    y_weak_train = Y_train_weak

    # Standardization with uk-dale values
    if model_ == 'solo_weakUK' or model_ == 'strong_weakUK' or model_=='mixed':
        train_mean = uk_params['mean']
        train_std =  uk_params['std']
    else:
        if model_=='solo_strongUK':
            train_mean = 273.93
            train_std = 382.70
        else:
            train_mean = refit_params['mean']
            train_std = refit_params['std']
    # print("STRONG-WEAK")
    # print(perc_strong)
    print("Mean train")
    print(train_mean)
    print("Std train")
    print(train_std)

    x_train = standardize_data(x_train,train_mean, train_std)
    X_val = standardize_data(X_val, train_mean, train_std)
    X_test = standardize_data(X_test,train_mean, train_std)

    batch_size = 64
    window_size = 2550
    drop = params[model_]['drop']
    kernel = params[model_]['kernel']
    num_layers = params[model_]['layers']
    gru_units = params[model_]['GRU']
    cs = params[model_]['cs']
    only_strong = params[model_]['no_weak']
    type_ = ''
    lr = 0.002
    weight= 1
    classes = 5

    pre_trained = params[model_]['pre_trained']
    CRNN = CRNN_construction(window_size,weight, lr=lr, classes=5, drop_out=drop, kernel = kernel, num_layers=num_layers, gru_units=gru_units, cs=cs,
                             path=pre_trained, only_strong=only_strong)
    CRNN.summary()
    if cs:
                early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_strong_level_final_custom_f1_score', mode='max',
                                                              patience=15, restore_best_weights=True)
    else:
                early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_strong_level_final_custom_f1_score', mode='max',
                                                              patience=15, restore_best_weights=True)
    if val_only_weak_uk or val_only_weak_re:
                early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_weak_level_custom_f1_score', mode='max',
                                                                  patience=15, restore_best_weights=True)


    log_dir_ = '/home/eprincipi/Weak_Supervision/weak_transfer_learning/transfer_models/logs/logs_CRNN'  + datetime.now().strftime("%Y%m%d-%H%M%S") + type_ + str(weight)
    tensorboard = TensorBoard(log_dir=log_dir_)
    file_writer = tf.summary.create_file_writer(log_dir_ + "/metrics")
    file_writer.set_as_default()

    if not test:
                    history = CRNN.fit(x=x_train, y=[y_strong_train, y_weak_train], shuffle=True, epochs=1000, batch_size=batch_size,
                                   validation_data=(X_val, [Y_val, Y_val_weak]), callbacks=[early_stop, tensorboard], verbose=1)
                    CRNN.save_weights(
                    '../transfer_models/CRNN_model_' + type_ + str(weight) + '_' + str(batch_size) + '.h5')

    else:
                    CRNN.load_weights('../fine_tuned_models/')

    output_strong, output_weak = CRNN.predict(x=X_val)
    output_strong_test_o, output_weak_test = CRNN.predict(x=X_test)
    output_strong_train, output_weak_train = CRNN.predict(x=X_train)
    print(Y_val.shape)
    print(output_strong.shape)

    shape = output_strong.shape[0] * output_strong.shape[1]
    shape_test = output_strong_test_o.shape[0] * output_strong_test_o.shape[1]
    shape_train = output_strong_train.shape[0] * output_strong_train.shape[1]


    Y_val = Y_val.reshape(shape, 5)
    Y_test = Y_test.reshape(shape_test, 5)
    Y_train = Y_train.reshape(shape_train, 5)
    output_strong = output_strong.reshape(shape, 5)
    output_strong_test = output_strong_test_o.reshape(shape_test, 5)
    output_strong_train = output_strong_train.reshape(shape_train, 5)

    if not (val_only_weak_uk or val_only_weak_re):
                        thres_strong = thres_analysis(Y_val, output_strong,classes)
    else:
        if model_ =='solo_strongUK':
            thres_strong = [0.62292016, 0.4462335, 0.5181894, 0.48374844, 0.46488932]
        else:
            thres_strong = [0.5, 0.5, 0.5, 0.5, 0.5]
    output_weak_test = output_weak_test.reshape(output_weak_test.shape[0] * output_weak_test.shape[1], 5)
    output_weak = output_weak.reshape(output_weak.shape[0] * output_weak.shape[1], 5)
    thres_weak = [0.501, 0.501, 0.501, 0.501, 0.501]

    assert (Y_val.shape == output_strong.shape)

    if model_ == 'strong_weakREFIT':
        thres_strong = [0.67614186, 0.4964771, 0.47968936, 0.39231053, 0.5259076]
    print("Estimated best thresholds:",thres_strong)

    output_strong_test = app_binarization_strong(output_strong_test, thres_strong, 5)
    output_strong_train = app_binarization_strong(output_strong_train, thres_strong,5)

    output_strong = app_binarization_strong(output_strong, thres_strong, 5)

    print("STRONG SCORES:")
    print("Validation")
    print(classification_report(Y_val, output_strong))
    print("Test")
    print(classification_report(Y_test, output_strong_test))
    print("Train")
    print(classification_report(Y_train, output_strong_train))

    