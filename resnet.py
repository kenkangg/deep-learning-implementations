from keras.layers import Input, Dense, MaxPooling2D, Add, AveragePooling2D, Flatten, Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras import optimizers
import keras

from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

def create_res_block(x):
    conv = Conv2D(int(x.shape[3]), (3, 3), padding='same')(x)
    normalize = BatchNormalization()(conv)
    relu = Activation('relu')(normalize)
    conv2 = Conv2D(int(conv.shape[3]), (3,3), padding='same')(relu)
    normalize = BatchNormalization()(conv)
    combine = Add()([normalize,x])
    relu = Activation('relu')(combine)
    return relu

def create_fully_connected(x, inputs):
    flatten = Flatten()(x)
    hidden = Dense(1000)(flatten)
    prediction = Dense(10, activation='softmax')(hidden)

    model = Model(inputs=inputs, outputs=prediction)
    optimizer = optimizers.SGD(lr=0.1, decay=1e-4, momentum=0.9)
    model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model





inputs = Input(shape=(32,32,3))

conv1 = Conv2D(64, (7, 7), strides=(2,2), padding='same')(inputs)
maxpool = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(conv1)
resblock = create_res_block(maxpool)
resblock = create_res_block(resblock)

dimension = 128

for _ in range(3):
    for _ in range(2):
        resblock = create_res_block(resblock)
    resblock = Conv2D(dimension, (1,1), padding='same')(resblock)
    dimension *= 2

avgpool = AveragePooling2D((3,3))(resblock)

model = create_fully_connected(avgpool, inputs)


model.fit(x_train, y_train, epochs=10)
model.evaluate(x=x_test, y=y_test)
