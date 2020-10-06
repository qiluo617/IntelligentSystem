from __future__ import division, print_function

import cv2
import numpy as np
import pickle as cPickle
from scipy.io import wavfile
from ai_f18_hw04 import *
from os import listdir
import random

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression


def load_image(path1,path2):
    data_set = []
    path_train_bee = path1
    fileList_bee = listdir(path_train_bee)
    if '.DS_Store' in fileList_bee:
        fileList_bee.remove('.DS_Store')
    for file in fileList_bee:
        fileName = path_train_bee + file + '/'
        filelist = listdir(fileName)
        if '.DS_Store' in filelist:
            filelist.remove('.DS_Store')
        for files in filelist:
            subset = [[], []]
            filename = fileName + files
            img = cv2.imread(filename)
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            scaled_gray_image = gray_image / 255.0
            subset[0] = np.reshape(scaled_gray_image, (1024, 1))
            subset[1] = np.array([[1], [0]])
            data_set.append(subset)

    path_train_noBee = path2
    fileList_noBee = listdir(path_train_noBee)
    if '.DS_Store' in fileList_noBee:
        fileList_noBee.remove('.DS_Store')
    for file in fileList_noBee:
        fileName = path_train_noBee + file + '/'
        filelist = listdir(fileName)
        if '.DS_Store' in filelist:
            filelist.remove('.DS_Store')
        for files in filelist:
            subset = [[], []]
            filename = fileName + files
            img = cv2.imread(filename)
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            scaled_gray_image = gray_image / 255.0
            subset[0] = np.reshape(scaled_gray_image, (1024, 1))
            subset[1] = np.array([[0], [1]])
            data_set.append(subset)

    random.shuffle(data_set)
    return data_set


def load_img_conv(path1,path2):
    data_set = []
    result_set = []
    path_train_bee = path1
    fileList_bee = listdir(path_train_bee)
    if '.DS_Store' in fileList_bee:
        fileList_bee.remove('.DS_Store')
    for file in fileList_bee:
        fileName = path_train_bee + file + '/'
        filelist = listdir(fileName)
        if '.DS_Store' in filelist:
            filelist.remove('.DS_Store')
        for files in filelist:
            # subset = [[], []]
            filename = fileName + files
            img = cv2.imread(filename)
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            scaled_gray_image = gray_image / 255.0
            image = np.reshape(scaled_gray_image, (-1, 32, 32, 1))
            data_set.append(image)
            result_set.append(np.array([1, 0]))

    path_train_noBee = path2
    fileList_noBee = listdir(path_train_noBee)
    if '.DS_Store' in fileList_noBee:
        fileList_noBee.remove('.DS_Store')
    for file in fileList_noBee:
        fileName = path_train_noBee + file + '/'
        filelist = listdir(fileName)
        if '.DS_Store' in filelist:
            filelist.remove('.DS_Store')
        for files in filelist:
            # subset = [[], []]
            filename = fileName + files
            img = cv2.imread(filename)
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            scaled_gray_image = gray_image / 255.0
            image = np.reshape(scaled_gray_image, (-1, 32, 32, 1))
            data_set.append(image)
            result_set.append(np.array([0, 1]))

    return [data_set, result_set]


def load_aud(path1, path2, path3):
    data_set = []

    path_train_bee = path1
    fileList_bee = listdir(path_train_bee)


    if '.DS_Store' in fileList_bee:
        fileList_bee.remove('.DS_Store')

    for files in fileList_bee:
        subset = [[], []]
        filename = path_train_bee + files
        samplerate, audio = wavfile.read(filename)
        data = audio / float(np.max(audio))

        # for i in range(0, len(data)):
        #     if data[i] < 0:
        #         count = count + 1

        # if len(data) != 88244:
        #     print(len(data))

        trim_data = data[:79380]
        test = (trim_data - np.min(trim_data))/(np.max(trim_data) - np.min(trim_data))

        # if len(data) == 88244:
        subset[0] = np.reshape(test, (len(test), 1))
        subset[1] = np.array([[1], [0], [0]])
        data_set.append(subset)

    path_train_cricket = path2
    fileList_cricket = listdir(path_train_cricket)

    if '.DS_Store' in fileList_cricket:
        fileList_cricket.remove('.DS_Store')

    for files in fileList_cricket:
        subset = [[], []]
        filename = path_train_cricket + files
        samplerate, audio = wavfile.read(filename)
        data = audio / float(np.max(audio))

        # if len(data) != 88244:
        #     print(len(data))

        trim_data = data[:79380]
        test = (trim_data - np.min(trim_data)) / (np.max(trim_data) - np.min(trim_data))

        # if len(data) == 88244:
        subset[0] = np.reshape(test, (len(test), 1))
        subset[1] = np.array([[0], [1], [0]])
        data_set.append(subset)

    path_train_noise = path3
    fileList_noise = listdir(path_train_noise)

    if '.DS_Store' in fileList_noise:
        fileList_noise.remove('.DS_Store')

    for files in fileList_noise:
        subset = [[], []]
        filename = path_train_noise + files
        samplerate, audio = wavfile.read(filename)
        data = audio / float(np.max(audio))

        # if len(data) != 88244:
        #     print(len(data))
        # if len(data) == 88244:
        trim_data = data[:79380]
        test = (trim_data - np.min(trim_data)) / (np.max(trim_data) - np.min(trim_data))

        # if len(data) == 88244:
        subset[0] = np.reshape(test, (len(test), 1))
        subset[1] = np.array([[0], [0], [1]])
        data_set.append(subset)

    random.shuffle(data_set)
    return data_set


def load_aud_conv(path1, path2, path3):
    data_set = []
    result_set = []
    path_train_bee = path1
    fileList_bee = listdir(path_train_bee)

    if '.DS_Store' in fileList_bee:
        fileList_bee.remove('.DS_Store')

    for files in fileList_bee:
        filename = path_train_bee + files
        samplerate, audio = wavfile.read(filename)
        data = audio / float(np.max(audio))

        # if len(data) != 88244:
        #     print(len(data))

        trim_data = data[:79380]
        aud = np.reshape(trim_data, [-1, 1, len(trim_data), 1])
        data_set.append(aud)
        result_set.append(np.array([1, 0, 0]))

    path_train_cricket = path2
    fileList_cricket = listdir(path_train_cricket)

    if '.DS_Store' in fileList_cricket:
        fileList_cricket.remove('.DS_Store')

    for files in fileList_cricket:
        filename = path_train_cricket + files
        samplerate, audio = wavfile.read(filename)
        data = audio / float(np.max(audio))

        # if len(data) != 88244:
        #     print(len(data))
        trim_data = data[:79380]
        aud = np.reshape(trim_data, [-1, 1, len(trim_data), 1])
        data_set.append(aud)
        result_set.append(np.array([0, 1, 0]))

    path_train_noise = path3
    fileList_noise = listdir(path_train_noise)

    if '.DS_Store' in fileList_noise:
        fileList_noise.remove('.DS_Store')

    for files in fileList_noise:
        filename = path_train_noise + files
        samplerate, audio = wavfile.read(filename)
        data = audio / float(np.max(audio))

        # if len(data) != 88244:
        #     print(len(data))
        trim_data = data[:79380]
        aud = np.reshape(trim_data, [-1, 1, len(trim_data), 1])
        data_set.append(aud)
        result_set.append(np.array([0, 0, 1]))

    return [data_set, result_set]


def build_tflearn_convnet_img():
    input_layer = input_data(shape=[None, 32, 32, 1])
    conv_layer1 = conv_2d(input_layer,
                         nb_filter=20,
                         filter_size=5,
                         activation='relu',
                         name='conv_layer_1')
    pool_layer1 = max_pool_2d(conv_layer1, 2, name='pool_layer_1')

    conv_layer2 = conv_2d(pool_layer1,
                          nb_filter=40,
                          filter_size=5,
                          activation='relu',
                          name='conv_layer_2')
    pool_layer2 = max_pool_2d(conv_layer2, 2, name='pool_layer_2')

    fc_layer_1 = fully_connected(pool_layer2, 2,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='fc_layer_2')
    network = regression(fc_layer_2, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.01)
    return tflearn.DNN(network)


def load_img_convnet_for_testing(path):
    input_layer = input_data(shape=[None, 32, 32, 1])
    conv_layer1 = conv_2d(input_layer,
                          nb_filter=20,
                          filter_size=5,
                          activation='relu',
                          name='conv_layer_1')
    pool_layer1 = max_pool_2d(conv_layer1, 2, name='pool_layer_1')

    conv_layer2 = conv_2d(pool_layer1,
                          nb_filter=40,
                          filter_size=5,
                          activation='relu',
                          name='conv_layer_2')
    pool_layer2 = max_pool_2d(conv_layer2, 2, name='pool_layer_2')

    fc_layer_1 = fully_connected(pool_layer2, 2,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2,
                                 activation='softmax',
                                 name='fc_layer_2')
    # build a model from the network.
    model = tflearn.DNN(fc_layer_2)
    # load the trained and persisted network.
    model.load(path)
    return model


def test_convnet_img(convnet_model, validX, validY):
    results = []
    for i in range(len(validX)):
        prediction = convnet_model.predict(validX[i].reshape([-1, 32, 32, 1]))
        results.append(np.argmax(prediction, axis=1)[0] == np.argmax(validY[i]))
    return sum((np.array(results) == True))/len(results)


def build_tflearn_convnet_aud():
    input_layer = input_data(shape=[None, 1, 79380, 1])
    conv_layer1 = conv_2d(input_layer,
                          nb_filter=20,
                          filter_size=5,
                          activation='relu',
                          name='conv_layer_1')
    pool_layer1 = max_pool_2d(conv_layer1, 2, name='pool_layer_1')

    conv_layer2 = conv_2d(pool_layer1,
                          nb_filter=40,
                          filter_size=5,
                          activation='relu',
                          name='conv_layer_2')
    pool_layer2 = max_pool_2d(conv_layer2, 2, name='pool_layer_2')

    fc_layer_1 = fully_connected(pool_layer2, 3,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 3,
                                 activation='softmax',
                                 name='fc_layer_2')
    network = regression(fc_layer_2, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.01)
    return tflearn.DNN(network)


def load_aud_convnet_for_testing(path):
    input_layer = input_data(shape=[None, 1, 79380, 1])
    conv_layer1 = conv_2d(input_layer,
                          nb_filter=20,
                          filter_size=5,
                          activation='relu',
                          name='conv_layer_1')
    pool_layer1 = max_pool_2d(conv_layer1, 2, name='pool_layer_1')

    conv_layer2 = conv_2d(pool_layer1,
                          nb_filter=40,
                          filter_size=5,
                          activation='relu',
                          name='conv_layer_2')
    pool_layer2 = max_pool_2d(conv_layer2, 2, name='pool_layer_2')

    fc_layer_1 = fully_connected(pool_layer2, 3,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 3,
                                 activation='softmax',
                                 name='fc_layer_2')
    # build a model from the network.
    model = tflearn.DNN(fc_layer_2)
    # load the trained and persisted network.
    model.load(path)
    return model


def test_convnet_aud(convnet_model, validX, validY):
    results = []
    for i in range(len(validX)):
        prediction = convnet_model.predict(validX[i].reshape([-1, 1, 79380, 1]))
        results.append(np.argmax(prediction, axis=1)[0] == np.argmax(validY[i]))
    return sum((np.array(results) == True)) / len(results)


def save(ann, file_name):
    with open(file_name, 'wb') as fp:
        cPickle.dump(ann, fp)


def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = cPickle.load(fp)
    return obj


image_bee_test_path = '/Users/qiluo/Desktop/AI/project1/BEE2Set/bee_test/'
image_Nobee_test_path = '/Users/qiluo/Desktop/AI/project1/BEE2Set/no_bee_test/'

image_bee_train_path = '/Users/qiluo/Desktop/AI/project1/BEE2Set/bee_train/'
image_Nobee_train_path = '/Users/qiluo/Desktop/AI/project1/BEE2Set/no_bee_train/'

audio_bee_test_path = '/Users/qiluo/Desktop/AI/project1/BUZZ2Set/test/bee_test/'
audio_cricket_test_path = '/Users/qiluo/Desktop/AI/project1/BUZZ2Set/test/cricket_test/'
audio_noise_test_path = '/Users/qiluo/Desktop/AI/project1/BUZZ2Set/test/noise_test/'

audio_bee_train_path = '/Users/qiluo/Desktop/AI/project1/BUZZ2Set/train/bee_train/'
audio_cricket_train_path = '/Users/qiluo/Desktop/AI/project1/BUZZ2Set/train/cricket_train/'
audio_noise_train_path = '/Users/qiluo/Desktop/AI/project1/BUZZ2Set/train/noise_train/'

#############IMAGE ANN###############

# bee_train_data = load_image(image_bee_train_path, image_Nobee_train_path)
# bee_test_data = load_image(image_bee_test_path,image_Nobee_test_path)
#
# net = Network([1024,300,2])
# net.SGD2(bee_train_data, 30, 10, 0.01, 5,bee_test_data,True,True,True,True)  #acc train: 35433 / 38139 test: 11753 / 12724
# save(net, 'ImageANN.pck')

#############AUDIO ANN###############

# audio_test = load_aud(audio_bee_test_path, audio_cricket_test_path, audio_noise_test_path)
# audio_train = load_aud(audio_bee_train_path, audio_cricket_train_path, audio_noise_train_path)
#
# net = Network([79380, 100, 3], cost=CrossEntropyCost)
# net.SGD2(audio_train, 10, 10, 0.01, 500, audio_test,True,True,True,True)
# # sizedown to 1323 [1323 1000 3]: 6721/7582 975/2332
# # [79380, 50, 3] : 5114 / 7582  1055 / 2332
# save(net, 'AudioANN.pck')

#############IMAGE CONV###############

# bee_train_data_conv = load_img_conv(image_bee_train_path, image_Nobee_train_path)
# bee_test_data_conv = load_img_conv(image_bee_test_path,image_Nobee_test_path)

# save(bee_test_data_conv[0], 'my_mnist_net/valid_x.pck')
# save(bee_test_data_conv[1], 'my_mnist_net/valid_y.pck')

# bee_train_data_conv[0], bee_train_data_conv[1] = shuffle(bee_train_data_conv[0], bee_train_data_conv[1])
# bee_test_data_conv[0], bee_test_data_conv[1] = shuffle(bee_test_data_conv[0], bee_test_data_conv[1])
#
# bee_train_data_conv[0] = bee_train_data_conv[0].reshape([-1, 32, 32, 1])
# bee_test_data_conv[0] = bee_test_data_conv[0].reshape([-1, 32, 32, 1])
#
# MODEL_img = build_tflearn_convnet_img()
#
# MODEL_img.fit(bee_train_data_conv[0], bee_train_data_conv[1], n_epoch=10,
#           shuffle=True,
#           validation_set=(bee_test_data_conv[0], bee_test_data_conv[1]),
#           # show_metric=True,
#           # snapshot_epoch=True,
#           snapshot_step=5000,
#           batch_size=10,
#           run_id='Image_ConvNet'
#           )
#
# MODEL_img.save('ImageConvNet.pck')
#
# print(MODEL_img.predict(bee_test_data_conv[0].reshape([-1, 32, 32, 1])))

#############AUDIO CONV###############

# audio_test_conv = load_aud_conv(audio_bee_test_path, audio_cricket_test_path, audio_noise_test_path)
# audio_train_conv = load_aud_conv(audio_bee_train_path, audio_cricket_train_path, audio_noise_train_path)
#
# audio_train_conv[0], audio_train_conv[1] = shuffle(audio_train_conv[0], audio_train_conv[1])
# audio_test_conv[0], audio_test_conv[1] = shuffle(audio_test_conv[0], audio_test_conv[1])
#
# audio_train_conv[0] = audio_train_conv[0].reshape([-1, 1, 79380, 1])
# audio_test_conv[0] = audio_test_conv[0].reshape([-1, 1, 79380, 1])
#
# MODEL_aud = build_tflearn_convnet_aud()
#
# MODEL_aud.fit(audio_train_conv[0], audio_train_conv[1], n_epoch=10,
#           # shuffle=True,
#           validation_set=(audio_test_conv[0], audio_test_conv[1]),
#           # show_metric=True,
#           # snapshot_epoch=True,
#           snapshot_step=5000,
#           batch_size=10,
#           run_id='Audio_ConvNet'
#           )
#
# MODEL_aud.save('AudioConvNet.pck')
#
# print(MODEL_aud.predict(audio_test_conv[0].reshape([-1, 1, 79380, 1])))


################TEST FOR CONV###################


#################IMAGE CONV TESTING######################

# bee_test_data_conv = load_img_conv(image_bee_test_path,image_Nobee_test_path)
# path_img_conv = '/Users/qiluo/Desktop/AI/project1/ImageConvNet.pck'
# model_conv_img = load_img_convnet_for_testing(path_img_conv)
# result_conv_img = test_convnet_img(model_conv_img, bee_test_data_conv[0], bee_test_data_conv[1])
# print(result_conv_img)

#################AUDIO CONV TESTING######################

# audio_test_conv = load_aud_conv(audio_bee_test_path, audio_cricket_test_path, audio_noise_test_path)
# path_aud_conv = '/Users/qiluo/Desktop/AI/project1/AudioConvNet.pck'
# model_conv_aud = load_aud_convnet_for_testing(path_aud_conv)
# result_conv_aud = test_convnet_aud(model_conv_aud, audio_test_conv[0], audio_test_conv[1])
# print(result_conv_aud)


