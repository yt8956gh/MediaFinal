# coding=utf-8

import os
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import optimizers
import tensorflow as tf
import keras.backend as K

'''
很重要!!!!!
GPU記憶體用量設定
'''
def GPU_usage():
    # number of GPU used
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # 設定GPU記憶體的使用比例
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.tensorflow_backend.set_session(sess)

'''
不同組別則更改成各自的function name
ex: style組:  def VGG16_style()
'''
def VGG16_category() :

    GPU_usage()
    # load training data from preprocessing.py
    train_data = np.load('./data/train_data.npy')
    train_label = np.load('./data/train_label.npy')
    test_data = np.load('./data/test_data.npy')
    test_label = np.load('./data/test_label.npy')

    # 使用keras VGG-16架構，並自行接上Fully-connect layer (Dense) 來進行分類
    # 最後一層Fully-connect layer參數量則為分類數量
    vgg16 = VGG16(include_top=False, weights='imagenet')
    x = vgg16.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(50, activation='relu')(x)
    predict = Dense(50, activation='softmax')(x)

    # Define input and output of the model
    model = Model(inputs=vgg16.input, outputs=predict)

    # Using Adam optimizer with lower learning rate
    # 針對不同問題可以使用不同的Loss function訓練
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    # Training the model for 5 epochs
    model.fit(
              x=train_data ,
              y=train_label,
              batch_size=16,
              epochs=3,
              validation_split=0.1,
              verbose=1
    )

    # 儲存訓練完成後的model權重，即完成model訓練
    # 當重新訓練後要更改檔名，否則會覆蓋掉之前train好的weights
    model.save_weights('./models/vgg_weights_pattern.h5')

    # testing model 需準備好test_data及test_label
    loss, accuracy = model.evaluate(x=test_data, y=test_label, batch_size=16, verbose=1)

    print("Testing: accuracy = %f  ;  loss = %f" % (accuracy, loss))


if __name__ == '__main__':
    VGG16_category()
