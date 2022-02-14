from tabnanny import verbose
import tensorflow as tf
from CNN import CNN
from detection_helper import get_all_files, load_balanced_data, model_acc

# gathering data
files = get_all_files()

# loading the balanced training data
X_train, y_train = load_balanced_data(files, 90000,0)
X_test, y_test = load_balanced_data(files, 20000, 150000)

# getting and compiling the model
model = CNN()
model.compile(optimizer = tf.keras.optimizers.SGD(1e-3, momentum=0.9), loss="binary_crossentropy", metrics = ['acc'])

# fitting the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 15, batch_size=256, verbose=1)

# plotting the accuracy of the models
model_acc(history)
