from keras.layers import Input, Dense, MaxPooling2D, Add, AveragePooling2D, Flatten, Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras import optimizers
import keras

class DeepResidualNetwork:
    """ Builds 18 layer Deep Residual Network

    This class only implements the 18-layer ResNet without bottlenecking
    residual blocks. Initializing this class will create the full network
    and the get_model() method should return the keras model.

    Args:
        input_shape (tuple): (width, height, channel)
        output_shape (int): number of classes

    """

    def __init__(self, input_shape, output_shape):
        inputs = Input(shape=input_shape)

        conv1 = Conv2D(64, (7, 7), strides=(2,2), padding='same')(inputs)
        normalize = BatchNormalization()(conv1)
        maxpool = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(normalize)
        resblock = self.create_res_block(maxpool)
        resblock = self.create_res_block(resblock)

        dimension = 128
        for _ in range(3):
            for _ in range(2):
                resblock = self.create_res_block(resblock)
            resblock = Conv2D(dimension, (1,1), padding='same')(resblock)
            dimension *= 2

        avgpool = AveragePooling2D((3,3))(resblock)

        self.model = self.create_fully_connected(avgpool, inputs, output_shape)


    def create_res_block(self, x):
        """ Creates a single Residual Block (Not Bottlenecked) """
        conv = Conv2D(int(x.shape[3]), (3, 3), padding='same')(x)
        normalize = BatchNormalization()(conv)
        relu = Activation('relu')(normalize)
        conv2 = Conv2D(int(conv.shape[3]), (3,3), padding='same')(relu)
        normalize = BatchNormalization()(conv)
        combine = Add()([normalize,x])
        relu = Activation('relu')(combine)
        return relu

    def create_fully_connected(self, x, inputs, output_shape):
        """ Creates a fully connected layer at the end of the residual blocks """
        flatten = Flatten()(x)
        # hidden = Dense(1000)(flatten)
        prediction = Dense(output_shape, activation='softmax')(flatten)

        model = Model(inputs=inputs, outputs=prediction)
        optimizer = optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9)
        model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        return model

    def get_model(self):
        """ Returns the initalized Keras model """
        return self.model
