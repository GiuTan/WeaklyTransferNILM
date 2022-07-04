import tensorflow as tf
from tensorflow.keras import backend as K
from pooling_layer import LinSoftmaxPooling1D
from losses import binary_crossentropy,binary_crossentropy_weak
from metrics import custom_f1_score


def CRNN_block(x, kernel,drop_out,filters):
    conv_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(kernel, 1), strides=(1, 1), padding='same',
                                    kernel_initializer='glorot_uniform')(x)
    print("conv_1")
    print(conv_1.shape)
    batch_norm_1 = tf.keras.layers.BatchNormalization()(conv_1)
    act_1 = tf.keras.layers.Activation('relu')(batch_norm_1)
    pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(1, 1))(act_1)
    drop_1 = tf.keras.layers.Dropout(drop_out)(pool_1)
    print("drop_1")
    print(drop_1.shape)
    return drop_1


def CRNN_construction(window_size, weight, lr=0.0, classes=0, drop_out = 0.1, kernel = 1, num_layers=1, gru_units=1, cs=False, path = '', only_strong=False):

    input_data = tf.keras.Input(shape=(window_size, 1))
    x = tf.keras.layers.Reshape((window_size,1,1))(input_data)

    for i in range(num_layers):
        filters = 2 ** (i+5)
        CRNN = CRNN_block(x, kernel=kernel, drop_out=drop_out, filters=filters)
        x = CRNN


    spec_x = tf.keras.layers.Reshape((x.shape[1], x.shape[3]))(x)
    print("Reshape")
    print(spec_x.shape)
    bi_direct = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=gru_units,return_sequences=True))(spec_x)
    print("bi direct")
    print(bi_direct.shape)
    frame_level = tf.keras.layers.Dense(units=classes, activation='sigmoid', name="strong_level")(bi_direct)
    print("frame level")
    print(frame_level.shape)
    pool_bag = LinSoftmaxPooling1D(axis=1)(frame_level)
    bag_level = tf.keras.layers.Activation('sigmoid', name="weak_level")(pool_bag)

    if cs:
            frame_level_final = tf.keras.layers.Multiply(name="strong_level_final")([bag_level, frame_level])
            print(frame_level_final.shape)


            model_CRNN = tf.keras.Model(inputs=input_data, outputs=[frame_level_final, bag_level], name="CRNN")
            model_CRNN.load_weights(path)


            conv_1 = model_CRNN.get_layer('conv2d')
            conv_1.trainable = False
            conv2d_1 = model_CRNN.get_layer('conv2d_1')
            conv2d_1.trainable = False
            conv2d_2 = model_CRNN.get_layer('conv2d_2')
            conv2d_2.trainable = False
            conv2d_3 = model_CRNN.get_layer('conv2d_3')
            conv2d_3.trainable = False
            # bidirectional = model_CRNN.get_layer('bidirectional')
            # bidirectional.trainable = False

            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

            model_CRNN.compile(optimizer=optimizer, loss={
                "strong_level_final": binary_crossentropy,
                "weak_level": binary_crossentropy_weak,
            }, metrics=[custom_f1_score], loss_weights=[1, weight])

    else:
        if only_strong:
            cs_s = False
            if cs_s:
                model_CRNN_ = tf.keras.Model(inputs=input_data, outputs=[frame_level], name="CRNN")
                model_CRNN_.load_weights(path)
                x = model_CRNN_.output
                pool_bag = LinSoftmaxPooling1D(axis=1)(x)
                bag_level = tf.keras.layers.Activation('sigmoid', name="weak_level")(pool_bag)
                frame_level_final = tf.keras.layers.Multiply(name="strong_level_final")([bag_level, frame_level])
                print(frame_level_final.shape)

                model_CRNN = tf.keras.Model(inputs=input_data, outputs=[frame_level_final, bag_level], name="CRNN")


                conv_1 = model_CRNN.get_layer('conv2d')
                conv_1.trainable = False
                conv2d_1 = model_CRNN.get_layer('conv2d_1')
                conv2d_1.trainable = False
                conv2d_2 = model_CRNN.get_layer('conv2d_2')
                conv2d_2.trainable = False
                # conv2d_3 = model_CRNN.get_layer('conv2d_3')
                # conv2d_3.trainable = False
                # bidirectional = model_CRNN.get_layer('bidirectional')
                # bidirectional.trainable = False

                optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

                model_CRNN.compile(optimizer=optimizer, loss={
                    "strong_level_final": binary_crossentropy,
                    "weak_level": binary_crossentropy_weak,
                }, metrics=[custom_f1_score], loss_weights=[1, weight])
            else:
                model_CRNN_ = tf.keras.Model(inputs=input_data, outputs=[frame_level], name="CRNN")
                model_CRNN_.load_weights(path)
                x = model_CRNN_.output
                pool_bag = LinSoftmaxPooling1D(axis=1)(x)
                bag_level = tf.keras.layers.Activation('sigmoid', name="weak_level")(pool_bag)
                model_CRNN = tf.keras.Model(inputs=model_CRNN_.input, outputs=[frame_level,bag_level])


                conv_1 = model_CRNN.get_layer('conv2d')
                conv_1.trainable = False
                conv2d_1 = model_CRNN.get_layer('conv2d_1')
                conv2d_1.trainable = False
                conv2d_2 = model_CRNN.get_layer('conv2d_2')
                conv2d_2.trainable = False
                # conv2d_3 = model_CRNN.get_layer('conv2d_3')
                # conv2d_3.trainable = False
                bidirectional = model_CRNN.get_layer('bidirectional')
                bidirectional.trainable = False

                optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

                model_CRNN.compile(optimizer=optimizer, loss={
                    "strong_level": binary_crossentropy,
                    "weak_level": binary_crossentropy_weak,
                }, metrics=[custom_f1_score], loss_weights=[1, weight])

        else:
            model_CRNN = tf.keras.Model(inputs=input_data, outputs=[frame_level, bag_level], name="CRNN")


            model_CRNN.load_weights(path)

            conv_1 = model_CRNN.get_layer('conv2d')
            conv_1.trainable = False
            conv2d_1 = model_CRNN.get_layer('conv2d_1')
            conv2d_1.trainable = False
            conv2d_2 = model_CRNN.get_layer('conv2d_2')
            conv2d_2.trainable = False
            # conv2d_3 = model_CRNN.get_layer('conv2d_3')
            # conv2d_3.trainable = False
            bidirectional = model_CRNN.get_layer('bidirectional')
            bidirectional.trainable = False
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

            model_CRNN.compile(optimizer=optimizer, loss={
                "strong_level": binary_crossentropy,
                "weak_level": binary_crossentropy_weak,
            }, metrics=[custom_f1_score], loss_weights=[1, weight])



    return model_CRNN

# layer = tf.keras.layers.Dense(3)
# layer.build((None, 4))  # Create the weights
# layer.trainable = True

# print("weights:", len(layer.weights))
# print("trainable_weights:", len(layer.trainable_weights))
# print("non_trainable_weights:", len(layer.non_trainable_weights))