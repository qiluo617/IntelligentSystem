from __future__ import division, print_function
import pickle as cPickle
from ai_f18_hw04 import *
import cv2
import numpy as np
from scipy.io import wavfile
from os import listdir
# from training_nets import load_img_convnet_for_testing, load_aud_convnet_for_testing


def load(file_name):
    with open(file_name, 'rb') as fp:
        nn = cPickle.load(fp)
    return nn


def fit_image_ann(ann, image_path):
    result = [0, 0]
    img = cv2.imread(image_path)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data = gray_image / 255.0
    data = np.reshape(data, (1024, 1))
    output = Network.feedforward(ann, data)
    # print(output)

    if output[0] >= 0.5:
        result[0] = 1
    else:
        result[0] = 0

    if output[1] >= 0.5:
        result[1] = 1
    else:
        result[1] = 0

    return result


def fit_image_convnet(convnet, image_path):
    result = [0, 0]
    img = cv2.imread(image_path)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data = gray_image / 255.0
    output = convnet.predict(data.reshape([-1, 32, 32, 1]))
    # print(output)

    if output[0][0] >= 0.5:
        result[0] = 1
    else:
        result[0] = 0

    if output[0][1] >= 0.5:
        result[1] = 1
    else:
        result[1] = 0

    return result


def fit_audio_ann(ann, audio_path):
    result = [0, 0, 0]
    samplerate, audio = wavfile.read(audio_path)
    data = audio / float(np.max(audio))
    trim_data = data[:79380]
    test = []
    i = 0
    while i < len(trim_data):
        test.append(trim_data[i])
        i = i + 60

    reshapeData = np.reshape(test, (len(test), 1))
    output = Network.feedforward(ann, reshapeData)
    print(output)

    if output[0] >= 0.5:
        result[0] = 1
    else:
        result[0] = 0

    if output[1] >= 0.5:
        result[1] = 1
    else:
        result[1] = 0

    if output[2] >= 0.5:
        result[2] = 1
    else:
        result[2] = 0

    return result


def fit_audio_convnet(convnet, audio_path):
    result = [0, 0, 0]
    samplerate, audio = wavfile.read(audio_path)
    data = audio / float(np.max(audio))
    trim_data = data[:79380]
    output = convnet.predict(trim_data.reshape([-1, 1, 79380, 1]))
    print(output)

    if output[0][0] >= 0.5:
        result[0] = 1
    else:
        result[0] = 0

    if output[0][1] >= 0.5:
        result[1] = 1
    else:
        result[1] = 0

    if output[0][2] >= 0.5:
        result[2] = 1
    else:
        result[2] = 0

    return result


# imageANN = load("/Users/qiluo/Desktop/AI/project1/ImageANN.pck")
# print(fit_image_ann(imageANN, '/Users/qiluo/Desktop/AI/project1/yb.png'))
# print(fit_image_ann(imageANN, '/Users/qiluo/Desktop/AI/project1/nb.png'))

# imageCON = load_img_convnet_for_testing('/Users/qiluo/Desktop/AI/project1/ImageConvNet.pck')
# print(fit_image_convnet(imageCON, '/Users/qiluo/Desktop/AI/project1/yb.png'))
# print(fit_image_convnet(imageCON, '/Users/qiluo/Desktop/AI/project1/nb.png'))


# imgANN = load("/Users/qiluo/Desktop/AI/project1/ImageANN.pck")
# path_img_bee = '/Users/qiluo/Desktop/AI/project1/BEE2Set/bee_test'  # test data: 5882 480 [92.455%]
# path_img_no_bee = '/Users/qiluo/Desktop/AI/project1/BEE2Set/no_bee_test' #test data: 506 5856 [95.034%]
#
# fileList_bee = listdir(path_img_no_bee)
#
# if '.DS_Store' in fileList_bee:
#     fileList_bee.remove('.DS_Store')
#
# sum = 0
# sum1 = 0
# for file in fileList_bee:
#     filename = path_img_no_bee +'/'+ file
#
#     fileList = listdir(filename)
#
#     if '.DS_Store' in fileList:
#         fileList.remove('.DS_Store')
#
#     for f in fileList:
#         i = filename + '/' + f
#         result = fit_image_ann(imgANN,i)
#         if np.argmax(result) == 0:
#             sum = sum +1
#         if np.argmax(result) == 1:
#             sum1 = sum1 +1
#
# print(sum)
# print(sum1)


# audioANN = load('/Users/qiluo/Desktop/AI/project1/AudioANN.pck')
# print(fit_audio_ann(audioANN, '/Users/qiluo/Desktop/AI/project1/bee.wav'))
# print(fit_audio_ann(audioANN, '/Users/qiluo/Desktop/AI/project1/noise.wav'))
# print(fit_audio_ann(audioANN, '/Users/qiluo/Desktop/AI/project1/cricket.wav'))

# audioCON = load_aud_convnet_for_testing('/Users/qiluo/Desktop/AI/project1/AudioConvNet.pck')
# print(fit_audio_convnet(audioCON, '/Users/qiluo/Desktop/AI/project1/bee.wav'))
# print(fit_audio_convnet(audioCON, '/Users/qiluo/Desktop/AI/project1/noise.wav'))
# print(fit_audio_convnet(audioCON, '/Users/qiluo/Desktop/AI/project1/cricket.wav'))

# audioANN = load('/Users/qiluo/Desktop/AI/project1/AudioANN_smallValid.pck')
# path_audio_bee = '/Users/qiluo/Desktop/AI/project1/BUZZ2Set/test/bee_test'
# # [1323_1000_3] test data: 1 0 897  #train data: 2151 216 35
# #[small valid] test:1 4 893
# path_audio_cricket = '/Users/qiluo/Desktop/AI/project1/BUZZ2Set/test/cricket_test'
# # [1323_1000_3] test data: 342 154 4  #train data: 806 2186 8
# #[small valid] test: 289 206 5
# path_audio_noise = '/Users/qiluo/Desktop/AI/project1/BUZZ2Set/test/noise_test'
# #[1323_1000_3] test data: 288 39 607  #train data: 143 25 2012
# #[small valid] test: 224 94 616
#
# fileList_bee = listdir(path_audio_bee)
#
# if '.DS_Store' in fileList_bee:
#     fileList_bee.remove('.DS_Store')
#
# sum = 0
# sum1 = 0
# sum2 = 0
# for file in fileList_bee:
#     filename = path_audio_bee +'/'+ file
#     result = fit_audio_ann(audioANN,filename)
#     if np.argmax(result) == 0:
#         sum = sum +1
#     if np.argmax(result) == 1:
#         sum1 = sum1 +1
#     if np.argmax(result) == 2:
#         sum2 = sum2 +1
#
# print(sum)
# print(sum1)
# print(sum2)

