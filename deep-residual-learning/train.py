import keras
from resnet import *
from keras.datasets import cifar10

num_classes = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

network = DeepResidualNetwork(x_train.shape[1:4], num_classes)
model = network.get_model()

model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))
