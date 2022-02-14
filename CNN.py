# creating our model
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization

class CNN(tf.keras.Model):
    def __init__(self, inp_shape=(50, 50, 3)):
        super().__init__()
        self.conv1 = Conv2D(32, (3,3), kernel_initializer='he_uniform', padding="same", activation='relu')
        self.max1 = MaxPooling2D(2)
        self.conv2 = Conv2D(64, (3,3), kernel_initializer='he_uniform', padding="same", activation='relu')
        self.max2 = MaxPooling2D(2)
        self.conv3 = Conv2D(128, (3,3), kernel_initializer='he_uniform', padding="same", activation='relu')
        self.max3 = MaxPooling2D(2)
        self.conv4 = Conv2D(128, (3,3), kernel_initializer='he_uniform', padding="same", activation='relu')
        self.max4 = MaxPooling2D(2)
        self.conv5 = Conv2D(256, (3,3), kernel_initializer='he_uniform', padding="same", activation='relu')
        self.max5 = MaxPooling2D(2)
        self.flattten = Flatten()
        self.dense1 = Dense(128, activation = 'relu')
        self.dense2 = Dense(1, activation ='sigmoid')

    def call(self, x):
        x = self.conv1(x)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.max2(x)
        x = self.conv3(x)
        x = self.max3(x)
        x = self.conv4(x)
        x = self.max4(x)
        x = self.conv5(x)
        x = self.max5(x)
        x = self.flattten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x