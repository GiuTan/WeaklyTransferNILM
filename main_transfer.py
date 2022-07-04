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
    path = '/home/eprincipi/Weak_Supervision/weak_labels/'
    file_agg_path =  path + 'dataset_weak/aggregate_data_noised/'
    file_labels_path = path + 'dataset_weak/labels/'
    model_ = 'strong_weakUK'

    # REFIT path
    WEAK_agg_resample_path = '/raid/users/eprincipi/resampled_agg_REFIT/'

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

    # perc_strong = 20
    # print("perc strong",perc_strong)
    # perc_weak = 100
    # print("perc weak:", perc_weak)

    # Flag Inizialization
    test = True

    strong = False
    strong_weak = True #se avessi voluto escludere da alcuni segmenti le labels weak
    test_ukdale = False
    weak_counter = True
    val_only_weak_uk = False
    validation_refit = False
    val_only_weak_re = False

    houses = [1,2,3,4,5]
    houses_id = [0, 'house_1/', 'house_2/', 'house_3/', 'house_4/', 'house_5/']

    #X_train, Y_train, Y_train_weak = [], [], []
    X_test, Y_test, Y_test_weak = [], [], []
    X_val, Y_val, Y_val_weak = [], [], []


    # # LOADING DATA FROM .JSON FOR LABELS STRONG AND WEAK AND .NPY FOR AGGREGATE   #

    #
    for k in houses:

        count_str = 0
        count_val = 0
        count_weak = 0
        count_STRONG_real = 0

        f = open(file_labels_path + 'labels_%d.json' % k)
        labels = json.load(f)
        print("Labels Loaded")
        if k == 1:
            quantity = quantity_1_u
        if k == 2:
            quantity = quantity_2_u
        if k == 5:
            quantity = quantity_5_u
        if k == 3:
            quantity = quantity_3_u
        if k == 4:
            quantity = quantity_4_u

        b = round(quantity / 5)
        a = round(b / 5)

        print("Aggregate Loading")
        for i in range(quantity):

            agg = np.load(file_agg_path + houses_id[k] + 'aggregate_%d.npy' %i)

            key = 'labels_%d' %i

            #  STRONG  #
            list_strong = labels[key]['strong']

            matrix = np.zeros((5, 2550))
            error_vectors = 0
            if len(agg) > 2550 or len(list_strong[0]) >2550 or  len(list_strong[1]) >2550 or len(list_strong[2]) >2550 or len(list_strong[3]) >2550 or len(list_strong[4]) >2550:

                continue

            else:

                for l in range(len(list_strong)):
                    matrix[l] = np.array(list_strong[l])


                if k == 1 or k == 5 or k==3 or k==4:

                    if i < a or (i>=b and i <(a+b)) or (i>=(b*2) and i<(b*2 + a)) or (i>= (b*3) and i<(b*3 + a)) or (i>= b*4 and i<(b*4 + a)):
                        # se rientra nei dati di validation rimangono strong


                                matrix = np.transpose(matrix)
                                X_val.append(agg)
                                Y_val.append(matrix)


                    else: #se va nei dati di train
                        print('Train ukdale')
                if k ==2:

                    matrix = np.transpose(matrix)
                    X_test.append(agg)
                    Y_test.append(matrix)

            ##### WEAK #####
            list_weak = labels[key]['weak']

            error_vectors = 0
            if len(agg) > 2550 or len(list_strong[0]) > 2550 or len(list_strong[1]) > 2550 or len(
                    list_strong[2]) > 2550 or len(list_strong[3]) > 2550 or len(list_strong[4]) > 2550:

                continue

            else:

                if k == 1 or k == 5 or k==3 or k==4:

                    if i < a or (i >= b and i < (a + b)) or (i >= (b * 2) and i < (b * 2 + a)) or (
                            i >= (b * 3) and i < (b * 3 + a)) or (i >= b * 4 and i < (b * 4 + a)):


                        Y_val_weak.append(np.array(list_weak).reshape(1,5))

                    else:
                        print('train ukdale weak')
                #         if strong_weak:  #
                #             num_data =  0 #round(quantity / 100 * (100 - percentage_with_weak))
                #             print(num_data)
                #             count_weak += 1
                #             if count_weak <=num_data:
                #                 list_weak = [-1,-1,-1,-1,-1]   #escludo le labels weak per una certa percentuale complementare a quella che segnalo (quindi di labels che uso)
                #                 Y_train_weak.append(np.array(list_weak).reshape(1, 5))
                #
                #             else:
                #                 Y_train_weak.append(np.array(list_weak).reshape(1,5))
                #
                #         else:
                #             Y_train_weak.append(np.array(list_weak).reshape(1, 5))
                if k == 2:
                     Y_test_weak.append(np.array(list_weak).reshape(1,5))

    # X_train = np.array(X_train)
    # Y_train = np.array(Y_train)
    # Y_train_weak = np.array(Y_train_weak)


    if not test_ukdale:
        X_test, Y_test, Y_test_weak = [], [], []
        X_train, Y_train, Y_train_weak = [], [], []
        agg_resample_path_test = '/raid/users/eprincipi/resampled_agg_REFIT_test/'
        labels_resample_path_test = '/raid/users/eprincipi/resampled_labels_REFIT_test/'
        houses = [2,4,8,9,15]


        for k in houses:

            count_str = 0
            count_val = 0
            count_weak = 0
            count_STRONG_real = 0

            quant = [0, 0, quantity_2_r, 0, quantity_4_r, 0, 0, 0, quantity_8_r, quantity_9_r, 0,0, 0, 0,0, quantity_15_r, 0, 0,0, 0,0,0, 0]

            for i in range(quant[k]):
                train_q = round(quant[k] * 30 / 100)

                agg = np.load(agg_resample_path_test + 'house_' + str(k) + '/aggregate_%d.npy' % i)
                labels_weak = np.load('/raid/users/eprincipi/clean_refit/dataset_weak/labels/' + 'house_' + str(k) + '/weak_labels_%d.npy' % i, allow_pickle=True)
                labels_strong = np.load(labels_resample_path_test + 'house_' + str(k) + '/strong_labels_%d.npy' % i, allow_pickle=True)


                error_vectors = 0
                if len(agg) > 2550 or labels_strong.shape[1] != 2550:

                    # Verifico che i segmenti di aggregato e le labels abbiano le dimensioni corrette. Durante la fase di creazione del dataset
                    # alcune attivazioni prese in maniera randomica sono anomale, e fanno si che il segmento non abbia le dimensioni richieste

                    continue

                else:
                    if i <= train_q:
                        matrix = np.transpose(labels_strong) #matrix = np.negative(np.ones((2550,5)))
                        X_train.append(agg)
                        Y_train.append(matrix)
                        Y_train_weak.append(labels_weak.reshape(1, 5))

                    else:
                        matrix = np.transpose(labels_strong)
                        X_test.append(agg)
                        Y_test.append(matrix)
                        Y_test_weak.append(labels_weak.reshape(1, 5))


        X_test = np.array(X_test)
        X_train = np.array(X_train)
        Y_test = np.array(Y_test)
        Y_train = np.array(Y_train)
        Y_train_weak = np.array(Y_train_weak)
        Y_test_weak =  np.array(Y_test_weak)
        print('Test shape', X_test)
        print('Train shape',X_train)
        # fine_tun_weak = True
        # if fine_tun_weak:
        #     Y_train = []
        #     matrix = np.negative(np.ones((2550,5)))
        #     for se in range(train_q):
        #         Y_train.append(matrix)
        #     Y_train = np.array(Y_train)
        # else:
        #     Y_train = np.array(Y_test[:train_q])
        # Y_test_weak = np.array(Y_test_weak[train_q:])
        # Y_train_weak = np.array(Y_test_weak[:train_q])
    else:
        print('Test ukdale')
        train_q = round(len(X_test)*30/100)
        y_weak = []
        matrix  = np.negative(np.ones((2550,5)))
        for nn in range(train_q):
            y_weak.append(matrix)
        X_train = np.array(X_test[:train_q])
        Y_train = np.array(Y_test[:train_q]) #np.array(y_weak)
        Y_train_weak = np.array(Y_test_weak[:train_q])
        assert(len(X_train)==len(Y_train))
        X_test = np.array(X_test[train_q:])
        Y_test = np.array(Y_test[train_q:])
        Y_test_weak = np.array(Y_test_weak[train_q:])
        Y_test = output_binarization(Y_test, 0.4)
        Y_train = output_binarization(Y_train, 0.4)

    Y_vall = []
    if val_only_weak_uk:
        for ll in range(len(Y_val)):
            matrix = np.ones((5, 2550))
            matrix = np.negative(matrix)
            matrix = np.transpose(matrix)
            Y_vall.append(matrix)

        assert(len(Y_vall)==len(Y_val))
        Y_val = np.array(Y_vall)
    if validation_refit:
       train_set, val_set, test_set = load_refit_data()
       del train_set
       del test_set
       gc.collect()

       X_val_ = val_set[0]
       Y_val_ = val_set[1]
       X_val = []
       Y_val = []
       for seg in range(len(X_val_)):
           time = pd.date_range('2014-01-01', periods=2550, freq='8s')
           agg1 = pd.Series(data=X_val_[seg], index=time)
           new_labels = []
           labels_strong = Y_val_[seg]
           for a in range(5):
               label = pd.Series(data=labels_strong[:,a], index=time)
               resampled_labels = label.resample('6s').bfill()
               new_labels.append(resampled_labels[:len(agg1)].to_numpy())
           resampled = agg1.resample('6s').bfill()
           new_labels = np.array(new_labels)
           new_labels = np.transpose(new_labels)
           X_val.append(resampled[:len(agg1)])
           Y_val.append(new_labels)


       Y_val_weak = val_set[2]
       if val_only_weak_re:
           matrix = np.ones((5, 2550))
           matrix = np.negative(matrix)
           matrix = np.transpose(matrix)
           for lll in range(len(val_set[1])):
                Y_vall.append(matrix)

           assert(len(Y_vall)==len(val_set[1]))
           Y_val = Y_vall
       # else:
       #     Y_val = val_set[1]


    X_val = np.array(X_val)
    Y_val = np.array(Y_val)
    Y_val_weak = np.array(Y_val_weak)
    if not (validation_refit or val_only_weak_uk):
        print('Control binarization validation ukdale')
        Y_val = output_binarization(Y_val, 0.4)

    assert(len(X_val)==len(Y_val))
    assert(len(Y_val)==len(Y_val_weak))

    houses_weak = [2,5,7,10,12,13,16]
    weak_X_train_balanced, weak_Y_train_balanced,weak_Y_train_weak_balanced   = [], [], []
    # todo only for mixed experiment
    # for k in houses_weak:
    #         quant = [0,0,20000,0,0,20000,0,20000,0,0,20000,0,15000,3000,0,0,5000]
    #
    #         count_str = 0
    #         count_val = 0
    #         count_weak = 0
    #         count_STRONG_real = 0
    #
    #         error_vectors = 0
    #         for i in range(quant[k]):
    #
    #             print("Aggregate Loading")
    #             agg = np.load(WEAK_agg_resample_path + 'house_' + str(k) + '/aggregate_%d.npy' % i)
    #
    #             labels_weak = np.load('/raid/users/eprincipi/clean_refit/dataset_WEAK/labels/' + 'house_' + str(k) + '/weak_labels_%d.npy' % i, allow_pickle=True)
    #             labels_strong = np.load('/raid/users/eprincipi/clean_refit/dataset_WEAK/labels/' + 'house_' + str(k) + '/strong_labels_%d.npy' % i, allow_pickle=True)
    #
    #             # La matrice di -1 serve ad indicare l'assenza di labels strong; questo valore verrà gestito opportunamente dalla loss in fase di train
    #             matrix = np.ones((5, 2550))
    #             matrix = np.negative(matrix)
    #             error_vectors = 0
    #
    #             if len(agg) > 2550 or labels_strong.shape[1] != 2550:
    #                 print("Errore")
    #                 error_vectors += 1
    #                 continue
    #
    #             else:
    #                 # I DATI SOLO WEAK VENGONO USATI SOLO IN TRAIN
    #                 print("TRAIN")
    #                 matrix = np.transpose(matrix)
    #                 weak_X_train_balanced.append(agg)
    #                 weak_Y_train_balanced.append(matrix)
    #                 weak_Y_train_weak_balanced.append(labels_weak.reshape(1, 5))
    #         print("Error vectors:", error_vectors)
    #
    # weak_X_train_balanced = np.array(weak_X_train_balanced)
    # weak_Y_train_balanced = np.array(weak_Y_train_balanced)
    # weak_Y_train_weak_balanced = np.array(weak_Y_train_weak_balanced)
    #
    # Y_train_new = []
    # X_train_new = []
    # Y_train_weak_new = []
    # # prendo solo la porzione strong e controllo che sia della numerosità che voglio
    # for i in range(len(Y_train)):
    #         if np.all(Y_train[i][0] != -1):
    #             Y_train_new.append(Y_train[i])
    #             X_train_new.append(X_train[i])
    #             Y_train_weak_new.append(Y_train_weak[i])
    #
    # Y_train_new = np.array(Y_train_new)
    # Y_train_weak_new = np.array(Y_train_weak_new)
    # X_train_new = np.array(X_train_new)
    # Y_train_new = output_binarization(Y_train_new, 0.4)
    # print("Y train strong shape", Y_train_new.shape)
    #
    # print("Train strong shape")
    # print(Y_train_new.shape)
    #
    # # UNISCO I DATI STRONG-WEAK CON QUELLI SOLO WEAK
    # X_train = weak_X_train_balanced #np.concatenate([X_train_new, weak_X_train_balanced], axis=0)
    # Y_train = weak_Y_train_balanced #np.concatenate([Y_train_new, weak_Y_train_balanced], axis=0)
    # Y_train_weak = weak_Y_train_weak_balanced #np.concatenate([Y_train_weak_new, weak_Y_train_weak_balanced], axis=0)
    #
    # num_weak = round(len(Y_train_weak) / 100 * perc_weak)
    # num_non_weak = round(len(Y_train_weak) / 100 * (100 - perc_weak))
    # Y_train_weak = Y_train_weak[:num_weak]
    # Y_train = Y_train[:num_weak]
    # X_train = X_train[:num_weak]
    #
    # # conto la numerosità di label weak considerata
    weak_count(Y_test_weak)

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
    type_ = 'UKpretrain_REFITfinetuningWeak_30_balanced0.01'#'solo_strongUK_pretrain_REFITfinetuning_WeakCS_BiDi_'
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
                    '/home/eprincipi/Weak_Supervision/weak_transfer_learning/transfer_models/CRNN_model_' + type_ + str(weight) + '_' + str(batch_size) + '.h5')

    else:
                    CRNN.load_weights('/home/eprincipi/Weak_Supervision/weak_transfer_learning/transfer_models/strong_weakUK/CRNN_model_UKpretrain_REFITfinetuningWeak_30_balanced0.01.h5')

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

    plt.plot(output_strong[:24000, 0])
    plt.plot(Y_val[:24000, 0])
    plt.legend(['output', 'strong labels'])
    plt.show()

    plt.plot(output_strong[:24000, 1])
    plt.plot(Y_val[:24000, 1])
    plt.legend(['output', 'strong labels'])
    plt.show()

    plt.plot(output_strong[:24000, 2])
    plt.plot(Y_val[:24000, 2])
    plt.legend(['output', 'strong labels'])
    plt.show()

    plt.plot(output_strong[:24000, 3])
    plt.plot(Y_val[:24000, 3])
    plt.legend(['output', 'strong labels'])
    plt.show()

    plt.plot(output_strong[:24000, 4])
    plt.plot(Y_val[:24000, 4])
    plt.legend(['output', 'strong labels'])
    plt.show()
    if model_ == 'strong_weakREFIT':
        thres_strong = [0.67614186, 0.4964771, 0.47968936, 0.39231053, 0.5259076]
    print("Estimated best thresholds:",thres_strong)

    output_strong_test = app_binarization_strong(output_strong_test, thres_strong, 5)
    output_strong_train = app_binarization_strong(output_strong_train, thres_strong,5)


    # np.save(
    #     '/home/eprincipi/Weak_Supervision/weak_labels/models_REFIT/output/transfer_learning/Y_test_transfer.npy',
    #     Y_test)

    np.save(
                 '/home/eprincipi/Weak_Supervision/weak_labels/models_REFIT/output/transfer_learning/' + type_ + '_test_prediction_bin.npy',
                output_strong_test)

    output_strong = app_binarization_strong(output_strong, thres_strong, 5)

    print("STRONG SCORES:")
    print("Validation")
    print(classification_report(Y_val, output_strong))
    print("Test")
    print(classification_report(Y_test, output_strong_test))
    print("Train")
    print(classification_report(Y_train, output_strong_train))

    if test_ukdale:
        houses = [2]
        X_test_synth = []
        Y_test = []
        file_agg_path_synth = path + 'dataset_weak/aggregate_data/'
        for k in houses:

            count_str = 0
            count_val = 0
            count_weak = 0
            count_STRONG_real = 0

            f = open(file_labels_path + 'labels_%d.json' % k)
            labels = json.load(f)
            print("Labels Loaded")
            if k == 2:
                quantity = quantity_2_u

            b = round(quantity / 5)
            a = round(b / 5)

            for i in range(quantity):

                agg = np.load(file_agg_path_synth + houses_id[k] + 'aggregate_%d.npy' % i)

                key = 'labels_%d' % i

                #  STRONG  #
                list_strong = labels[key]['strong']

                matrix_ = np.zeros((5, 2550))
                error_vectors = 0
                if len(agg) > 2550 or len(list_strong[0]) > 2550 or len(list_strong[1]) > 2550 or len(
                                            list_strong[2]) > 2550 or len(list_strong[3]) > 2550 or len(list_strong[4]) > 2550:

                                    continue

                else:

                    #matrix = np.transpose(matrix)
                    X_test_synth.append(agg)
                    # Y_test.append(matrix_)

            train_q = round(len(X_test_synth) * 30 / 100)
            print(train_q)
            X_test_synth = np.array(X_test_synth[train_q:])

            ANE = ANE(X_test_synth, output_strong_test_o)
            print("ANE UKDALE:",ANE)

    else:
        synth_agg_path = '/raid/users/eprincipi/resampled_agg_synth_REFIT_test/'
        file_labels_path = '/raid/users/eprincipi/resampled_labels_REFIT_test/'
        houses_re_test = [2, 4, 8, 9, 15]

        X_test = []
        Y_test = []

        for k in houses_re_test:

            count_str = 0
            count_val = 0
            count_weak = 0
            count_STRONG_real = 0

            quant = [0, 0, quantity_2_r, 0, quantity_4_r, 0, 0, 0,quantity_8_r, quantity_9_r, 0, 0, 0, 0, 0, quantity_15_r,0, 0, 0, 0]

            for i in range(quant[k]):
                train_q = round(quant[k] * 30 / 100)


                agg = np.load(synth_agg_path + 'house_' + str(k) + '/aggregate_%d.npy' % i)

                labels_strong = np.load(
                                        file_labels_path + 'house_' + str(k) + '/strong_labels_%d.npy' % i,
                                        allow_pickle=True)

                if len(agg) > 2550 or labels_strong.shape[1] != 2550:

                    continue


                else:

                    if i <= train_q:

                        matrix = np.negative(np.ones((2550, 5)))

                    else:

                        X_test.append(agg)



        X_test_synth = np.array(X_test)
        print(X_test_synth.shape)
        print(output_strong_test_o.shape)
        assert(X_test_synth.shape[0]==output_strong_test_o.shape[0])
        ANE = ANE(X_test_synth, output_strong_test_o)
        print("ANE REFIT:",ANE)

