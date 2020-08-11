# Lint as: python3
"""Models for CERN."""
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l1
import h5py
from tensorflow.keras.constraints import *
from qkeras import *

def dense_model(Inputs, nclasses, l1Reg=0, dropoutRate=0.25):
    """
    Dense matrix, defaults similar to 2016 DeepCSV training
    """
    x = Dense(
        200, activation='relu', kernel_initializer='lecun_uniform',
        name='fc1_relu', kernel_regularizer=l1(l1Reg))(Inputs)
    x = Dropout(dropoutRate)(x)
    x = Dense(
        200, activation='relu', kernel_initializer='lecun_uniform',
        name='fc2_relu', kernel_regularizer=l1(l1Reg))(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(
        200, activation='relu', kernel_initializer='lecun_uniform',
        name='fc3_relu', kernel_regularizer=l1(l1Reg))(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(
        200, activation='relu', kernel_initializer='lecun_uniform',
        name='fc4_relu', kernel_regularizer=l1(l1Reg))(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(
        200, activation='relu', kernel_initializer='lecun_uniform',
        name='fc5_relu', kernel_regularizer=l1(l1Reg))(x)
    predictions = Dense(
        nclasses, activation='softmax', kernel_initializer='lecun_uniform',
        name = 'output_softmax')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def two_layer_model(Inputs, nclasses, l1Reg=0):
    """
    One hidden layer model
    """
    x = Dense(
        32, activation='relu', kernel_initializer='lecun_uniform',
        name='fc1_relu', kernel_regularizer=l1(l1Reg))(Inputs)
    predictions = Dense(
        nclasses, activation='sigmoid', kernel_initializer='lecun_uniform',
        name = 'output_sigmoid', kernel_regularizer=l1(l1Reg))(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def three_layer_model(Inputs, nclasses, l1Reg=0):
    """
    Two hidden layers model
    """
    x = Dense(64, activation='relu', kernel_initializer='lecun_uniform',
              name='fc1_relu', kernel_regularizer=l1(l1Reg))(Inputs)
    x = Dense(32, activation='relu', kernel_initializer='lecun_uniform',
              name='fc2_relu', kernel_regularizer=l1(l1Reg))(x)
    x = Dense(32, activation='relu', kernel_initializer='lecun_uniform',
              name='fc3_relu', kernel_regularizer=l1(l1Reg))(x)
    predictions = Dense(
        nclasses, activation='softmax', kernel_initializer='lecun_uniform',
        name='output_softmax', kernel_regularizer=l1(l1Reg))(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def three_layer_model_batch_norm(Inputs, nclasses, l1Reg=0):
    """
    Two hidden layers model
    """
    x = Dense(64, kernel_initializer='he_uniform', use_bias=False,
              name='fc1_relu', kernel_regularizer=l1(l1Reg))(Inputs)
    x = BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn1')(x)
    x = Activation(activation='relu', name='relu1')(x)
    x = Dense(32, kernel_initializer='he_uniform', use_bias=False,
              name='fc2_relu', kernel_regularizer=l1(l1Reg))(x)
    x = BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn2')(x)
    x = Activation(activation='relu', name='relu2')(x)
    x = Dense(32, kernel_initializer='he_uniform', use_bias=False,
              name='fc3_relu', kernel_regularizer=l1(l1Reg))(x)
    x = BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn3')(x)
    x = Activation(activation='relu', name='relu3')(x)
    x = Dense(nclasses, kernel_initializer='he_uniform', use_bias=False,
              name='output_softmax', kernel_regularizer=l1(l1Reg))(x)
    #x = BatchNormalization(epsilon=1e-6, momentum=0.9, name='bn4')(x)
    predictions = Activation(activation='softmax', name='softmax')(x)

    model = Model(inputs=Inputs, outputs=predictions)
    return model


def three_layer_model_tanh(Inputs, nclasses, l1Reg=0):
    """
    Two hidden layers model
    """
    x = Dense(64, activation='tanh', kernel_initializer='lecun_uniform',
              name='fc1_tanh', kernel_regularizer=l1(l1Reg))(Inputs)
    x = Dense(32, activation='tanh', kernel_initializer='lecun_uniform',
              name='fc2_tanh', kernel_regularizer=l1(l1Reg))(x)
    x = Dense(32, activation='tanh', kernel_initializer='lecun_uniform',
              name='fc3_tanh', kernel_regularizer=l1(l1Reg))(x)
    predictions = Dense(
        nclasses, activation='softmax', kernel_initializer='lecun_uniform',
        name='output_softmax', kernel_regularizer=l1(l1Reg))(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def linear_model(Inputs, nclasses, l1Reg=0):
    """
    Linear model
    """
    predictions = Dense(
        nclasses, activation='linear', kernel_initializer='lecun_uniform',
        name='output_linear')(Inputs)
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def conv1d_model(Inputs, nclasses, l1Reg=0):
    """
    Conv1D model, kernel size 40
    """
    x = Conv1D(filters=32, kernel_size=40, strides=1, padding='same',
               kernel_initializer='he_normal', use_bias=True, name='conv1_relu',
               activation = 'relu', kernel_regularizer=l1(l1Reg))(Inputs)
    x = Conv1D(filters=32, kernel_size=40, strides=1, padding='same',
               kernel_initializer='he_normal', use_bias=True, name='conv2_relu',
               activation = 'relu', kernel_regularizer=l1(l1Reg))(x)
    x = Conv1D(filters=32, kernel_size=40, strides=1, padding='same',
               kernel_initializer='he_normal', use_bias=True, name='conv3_relu',
               activation = 'relu', kernel_regularizer=l1(l1Reg))(x)
    x = Flatten()(x)
    x = Dense(50, activation='relu', kernel_initializer='lecun_uniform',
              name='fc1_relu', kernel_regularizer=l1(l1Reg))(x)
    predictions = Dense(nclasses, activation='softmax', kernel_initializer='lecun_uniform',
                        name='output_softmax', kernel_regularizer=l1(l1Reg))(x)
    model = Model(inputs=Inputs, outputs=predictions)
    print(model.summary())
    return model

def conv1d_small_model(Inputs, nclasses, l1Reg=0):
    """
    Conv1D small model, kernel size 4
    """
    x = Conv1D(filters=3, kernel_size=4, strides=1, padding='same',
               kernel_initializer='he_normal', use_bias=True, name='conv1',
               kernel_regularizer=l1(l1Reg))(Inputs)
    x = Activation("relu", name="conv1_relu")(x)
    x = Conv1D(filters=2, kernel_size=4, strides=2, padding='same',
               kernel_initializer='he_normal', use_bias=True, name='conv2',
               activation = 'relu', kernel_regularizer=l1(l1Reg))(x)
    x = Activation("relu", name="conv2_relu")(x)
    x = Conv1D(filters=1, kernel_size=4, strides=3, padding='same',
               kernel_initializer='he_normal', use_bias=True, name='conv3',
               activation = 'relu', kernel_regularizer=l1(l1Reg))(x)
    x = Activation("relu", name="conv3_relu")(x)
    x = Flatten()(x)
    x = Dense(5, activation='relu', kernel_initializer='lecun_uniform',
              name='fc1_relu', kernel_regularizer=l1(l1Reg))(x)
    predictions = Dense(
        nclasses, activation='softmax', kernel_initializer='lecun_uniform',
        name='output_softmax', kernel_regularizer=l1(l1Reg))(x)
    model = Model(inputs=Inputs, outputs=predictions)
    print(model.summary())
    return model


def conv2d_model(Inputs, nclasses, l1Reg=0):
    """
    Conv2D model, kernel size (11,11), (3,3), (3,3)
    """
    x = Conv2D(filters=8, kernel_size=(11,11), strides=(1,1), padding='same',
               kernel_initializer='he_normal', use_bias=True, name='conv1_relu',
               activation = 'relu')(Inputs)
    x = Conv2D(filters=4, kernel_size=(3,3), strides=(2,2), padding='same',
               kernel_initializer='he_normal', use_bias=True, name='conv2_relu',
               activation = 'relu')(x)
    x = Conv2D(filters=2, kernel_size=(3,3), strides=(2,2), padding='same',
               kernel_initializer='he_normal', use_bias=True, name='conv3_relu',
               activation = 'relu')(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    predictions = Dense(
        nclasses, activation='softmax', kernel_initializer='lecun_uniform',
        name='output_softmax')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    print(model.summary())
    return model

if __name__ == '__main__':
    print(conv1d_model(Input(shape=(100,10,)), 2).summary())
    print(conv2d_model(Input(shape=(10,10,3,)), 2).summary())
