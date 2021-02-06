from keras.layers import *
from keras.layers import Reshape, Lambda, concatenate
from keras.models import Model, Sequential
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf
import keras.backend as K
import keras
# import tensorflow_graphics as tfg


class MatMul(Layer):

    def __init__(self, **kwargs):
        super(MatMul, self).__init__(**kwargs)

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list):
            raise ValueError('`MatMul` layer should be called '
                             'on a list of inputs')
        if len(input_shape) != 2:
            raise ValueError('The input of `MatMul` layer should be a list containing 2 elements')

        if len(input_shape[0]) != 3 or len(input_shape[1]) != 3:
            raise ValueError('The dimensions of each element of inputs should be 3')

        if input_shape[0][-1] != input_shape[1][1]:
            raise ValueError('The last dimension of inputs[0] should match the dimension 1 of inputs[1]')

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('A `MatMul` layer should be called '
                             'on a list of inputs.')
        return tf.matmul(inputs[0], inputs[1])

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0][0], input_shape[0][1], input_shape[1][-1]]
        return tuple(output_shape)


class MaskRelu(Layer):
    def __init__(self, **kwargs):
        super(MaskRelu, self).__init__(**kwargs)

    def build(self, input_shape):
        kernel_shape = input_shape

    def call(self, x, mask=None):
        t = 0.2
        x = K.sigmoid(x)
        inv_msk = K.relu(x - t, max_value=1)
        return inv_msk


class KLDivergenceLayer(Layer):
    """ Identity transform layer that adds KL divergence
        to the final model loss.
        """

    def __init__(self, beta, *args, **kwargs):
        self.is_placeholder = True
        self.beta = beta
        # super(KLDivergenceLayer, self).__init__(*args, **kwargs)
        super(KLDivergenceLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        mu, log_var = inputs
        kl_batch = - 0.5 * K.sum(1. + log_var - K.square(mu) - K.exp(log_var), axis=-1)
        self.add_loss(self.beta * K.mean(kl_batch), inputs=inputs)
        return inputs


def point_mask(nb_classes):
    input_points = Input(shape=(2048, 3))
    x = Conv1D(64, 1, activation='relu')(input_points)
    x = BatchNormalization()(x)
    x = Conv1D(128, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(1024, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2048)(x)

    z_mu = Conv1D(2048, 1)(x)
    z_log_var = Conv1D(2048, 1)(x)
    z_mu, z_log_var = KLDivergenceLayer(0.5)([z_mu, z_log_var])
    z_sigma = Lambda(lambda t: K.exp(0.5 * t))(z_log_var)
    eps = Input(tensor=K.random_normal(stddev=1.0, shape=(K.shape(input_points)[0], 1, 2048)), name='eps_input')
    z_eps = Multiply()([z_sigma, eps])
    z = Add()([z_mu, z_eps])
    z = MaskRelu(name='mask')(z)
    z = Permute((2, 1))(z)
    masked_input = multiply([input_points, z], name='modified_in')

    x = Conv1D(64, 1, activation='relu')(masked_input)
    x = BatchNormalization()(x)
    x = Conv1D(128, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(1024, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2048)(x)

    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)

    x = Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(x)
    input_T = Reshape((3, 3))(x)

    # forward net
    g = MatMul(name='rotated_in')([input_points, input_T])
    g = Conv1D(64, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Conv1D(64, 1, activation='relu')(g)
    g = BatchNormalization()(g)

    # feature transform net
    f = Conv1D(64, 1, activation='relu')(g)
    f = BatchNormalization()(f)
    f = Conv1D(128, 1, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Conv1D(1024, 1, activation='relu')(f)
    f = BatchNormalization()(f)
    f = MaxPooling1D(pool_size=2048)(f)
    f = Dense(512, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Dense(256, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(f)
    feature_T = Reshape((64, 64))(f)

    # forward net
    g = MatMul()([g, feature_T])
    g = Conv1D(64, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Conv1D(128, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Conv1D(1024, 1, activation='relu')(g)
    g = BatchNormalization()(g)

    # global feature
    global_feature = MaxPooling1D(pool_size=2048)(g)

    # point_net_cls
    c = Dense(512, activation='relu')(global_feature)
    c = BatchNormalization()(c)
    # c = Dropout(0.5)(c)
    c = Dense(256, activation='relu')(c)
    c = BatchNormalization()(c)
    # c = Dropout(0.5)(c)
    c = Dense(nb_classes, activation='softmax')(c)
    prediction = Flatten()(c)

    model = Model(inputs=[input_points, eps], outputs=prediction)

    return model
