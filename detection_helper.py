from glob import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os


# output all of the files in given directory
def get_all_files(dir='kaggle/*/*/*'):
    return glob(dir)


# displays 25 random images in all of the data
def show_img(files):
    plt.figure(figsize= (10,10))
    ind = np.random.randint(0, len(files), 25)
    i=0
    for loc in ind:
        plt.subplot(5,5,i+1)
        sample = load_img(files[loc], target_size=(150,150))
        sample = img_to_array(sample)
        plt.axis("off")
        plt.imshow(sample.astype("uint8"))
        i+=1


# loading unbalanced data
# load all of the data as (PIL Image, int)
def load_data(files, lower_limit, upper_limit):
    X = []
    y = []
    ind = 0
    for file in files[lower_limit:upper_limit]:
        if file.endswith(".png"):
            img = load_img(file, target_size = (50,50))
            pixels = img_to_array(img)
            pixels /= 255
            X.append(pixels)
            if(file[-5] == '1'):
                y.append(1)
            elif(file[-5] == '0'):
                y.append(0)
        if ind % 1000 == 0:
            print('element:', ind)
        ind += 1
    # np.stack is essentially a way of joining arrays
    return np.array(np.stack(X)), np.array(y)


# loading the data but in a balanced way
def load_balanced_data(files, size, start_index):
    half_size = int(size/2)
    count=0
    res = []
    y = []
    # appending all the values with 1 first
    for file in files[start_index:]:
        if (count!=half_size):
            if file[-5] == '1' and file.endswith(".png"):
                img = load_img(file, target_size = (50,50))
                pixels = img_to_array(img)
                pixels /= 255
                res.append(pixels)
                y.append(1)
                count += 1

    # appending all of the values with 0
    for file in files[start_index:]:
        if(count!=0):
            if(file[-5] == '0'):
                img = load_img(file, target_size = (50,50))
                pixels = img_to_array(img)
                pixels /= 255
                res.append(pixels)
                y.append(0)
                count -= 1
    return np.array(np.stack(res)), np.array(y)


# plotting model accuracy
def model_acc(history):
    plt.figure(figsize = (12,6))
    plt.subplot(2,1,1)
    plt.plot(history.history['acc'], label="train_acc")
    plt.plot(history.history['val_acc'], label = "test_acc")
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(history.history['loss'], label = "train_loss")
    plt.plot(history.history['val_loss'], label = "val_loss")
    plt.legend()
