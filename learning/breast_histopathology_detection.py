import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
import os
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

# creating the CNN model
# CNN model
def CNN():
    # model
    model = Sequential()
    model.add(Conv2D(16, (5, 5), strides=(1,1), activation='relu', input_shape=(50,50,3)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(16, (5, 5), strides=(1,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# load all variables into X and y
X = os.listdir('data/0and1/')
y = []
for x in X:
    if x[0] == '0':
        y.append(0)
    else:
        y.append(1)

# splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X = []
y = []

print('data is split')

# putting values into X_train, ... as a np.array
# X_train = [list(Image.open('data/0and1/' + i).getdata()) for i in X_train]
X_train_numpy = []
for i in X_train:
    im = Image.open('data/0and1/' + i)
    arr = np.array(im).astype('float32').tolist()
    X_train_numpy.append(arr)
    im.close()
X_train = np.array(X_train_numpy)
X_train_numpy = []
print('done1')

y_train = np.array(y_train)
print('done2')

# with open('saved_arr/train.npy', 'wb') as f:
#     np.save(f, X_train)
#     np.save(f, y_train)

# X_test = [list(Image.open('data/0and1/' + i).getdata()) for i in X_test]
X_test_numpy = []
for i in X_test:
    im = Image.open('data/0and1/' + i)
    arr = np.array(im).astype('float32').tolist()
    X_test_numpy.append(arr)
    im.close()
X_test = np.array(X_test_numpy)
X_test_numpy = []
print('done3')

y_test = np.array(y_test)
print('done4')

# # save into file
# with open('saved_arr/test.npy', 'wb') as f:
#     np.save(f, X_test)
#     np.save(f, y_test)

print('data is loaded fully')
print('data is saved')

# building the model
model = CNN()
print(model.summary())

# fitting the model
model.fit(X_train, y_train, epochs=10, batch_size=200, verbose=1)

# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: {} \n Error: {}".format(scores[1], 100-scores[1]*100))

model.save('model/my_model')
