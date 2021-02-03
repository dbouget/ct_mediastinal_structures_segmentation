from tensorflow.python.keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Dropout, Flatten, SpatialDropout2D, \
    ZeroPadding2D, Activation, AveragePooling2D, UpSampling2D, BatchNormalization, ConvLSTM2D, \
    TimeDistributed, Concatenate, Lambda, Reshape, UpSampling3D, Convolution3D, MaxPooling3D, SpatialDropout3D,\
    Conv2DTranspose, Conv3DTranspose, add, multiply, Reshape, Softmax, AveragePooling3D, Add, Layer
from tensorflow.python.keras.models import Model
import tensorflow as tf
import numpy as np


def convolution_block(x, nr_of_convolutions, use_bn=False, spatial_dropout=None):
    for i in range(2):
        x = Convolution3D(nr_of_convolutions, 3, padding='same')(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if spatial_dropout:
            x = SpatialDropout3D(spatial_dropout)(x)

    return x


def attention_block(g, x, nr_of_convolutions):
    """
    Taken from https://github.com/LeeJunHyun/Image_Segmentation
    """
    g1 = Convolution3D(nr_of_convolutions, kernel_size=1, strides=1, padding='same', use_bias=True)(g)
    g1 = BatchNormalization()(g1)

    x1 = Convolution3D(nr_of_convolutions, kernel_size=1, strides=1, padding='same', use_bias=True)(x)
    x1 = BatchNormalization()(x1)

    psi = Concatenate()([g1, x1])
    psi = Activation(activation='relu')(psi)
    psi = Convolution3D(1, kernel_size=1, strides=1, padding='same', use_bias=True)(psi)
    psi = BatchNormalization()(psi)
    psi = Activation(activation='sigmoid')(psi)

    return multiply([x, psi])


def attention_block_oktay(g, x, nr_of_convolutions):
    """
    Following the original paper and implementation at https://github.com/ozan-oktay/Attention-Gated-Networks
    """
    g1 = Convolution3D(nr_of_convolutions, kernel_size=1, strides=1, padding='same', use_bias=True)(g)
    g1 = BatchNormalization()(g1)

    x1 = MaxPooling3D([2, 2, 2])(x)
    x1 = Convolution3D(nr_of_convolutions, kernel_size=1, strides=1, padding='same', use_bias=True)(x1)
    x1 = BatchNormalization()(x1)

    psi = Concatenate()([g1, x1])
    psi = Activation(activation='relu')(psi)
    psi = Convolution3D(1, kernel_size=1, strides=1, padding='same', use_bias=True)(psi)
    psi = BatchNormalization()(psi)
    psi = Activation(activation='sigmoid')(psi)

    return multiply([x, psi])


def encoder_block(x, nr_of_convolutions, use_bn=False, spatial_dropout=None):
    x_before_downsampling = convolution_block(x, nr_of_convolutions, use_bn, spatial_dropout)
    downsample = [2, 2, 2]
    for i in range(1, 4):
        if x.shape[i] <= 4:
            downsample[i-1] = 1

    x = MaxPooling3D(downsample)(x_before_downsampling)

    return x, x_before_downsampling


def encoder_block_pyramid(x, input_ds, nr_of_convolutions, use_bn=False, spatial_dropout=None):
    pyramid_conv = Convolution3D(filters=nr_of_convolutions, kernel_size=(3, 3, 3), padding='same', activation='relu')(input_ds)
    x = Concatenate(axis=-1)([pyramid_conv, x])
    x_before_downsampling = convolution_block(x, nr_of_convolutions, use_bn, spatial_dropout)
    downsample = [2, 2, 2]
    for i in range(1, 4):
        if x.shape[i] <= 4:
            downsample[i-1] = 1

    x = MaxPooling3D(downsample)(x_before_downsampling)

    return x, x_before_downsampling


def decoder_block(x, cross_over_connection, nr_of_convolutions, use_bn=False, spatial_dropout=None):
    x = Conv3DTranspose(nr_of_convolutions, kernel_size=3, padding='same', strides=2)(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    attention = attention_block(g=x, x=cross_over_connection, nr_of_convolutions=int(nr_of_convolutions/2))
    x = Concatenate()([x, attention])
    x = convolution_block(x, nr_of_convolutions, use_bn, spatial_dropout)

    return x


class AttentionGatedUnet():
    def __init__(self, input_shape, nb_classes, deep_supervision=False, input_pyramid=False):
        if len(input_shape) != 3 and len(input_shape) != 4:
            raise ValueError('Input shape must have 3 or 4 dimensions')
        if nb_classes <= 1:
            raise ValueError('Segmentation classes must be > 1')
        self.dims = 3
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.deep_supervision = deep_supervision
        self.input_pyramid = input_pyramid
        self.convolutions = None
        self.encoder_use_bn = True
        self.decoder_use_bn = True
        self.encoder_spatial_dropout = None
        self.decoder_spatial_dropout = None

    def set_convolutions(self, convolutions):
        self.convolutions = convolutions

    def get_dice_loss(self):
        def dice_loss(target, output, epsilon=1e-10):
            smooth = 1.
            dice = 0

            for object in range(0, self.nb_classes):
                if self.dims == 2:
                    output1 = output[:, :, :, object]
                    target1 = target[:, :, :, object]
                else:
                    output1 = output[:, :, :, :, object]
                    target1 = target[:, :, :, :, object]
                intersection1 = tf.reduce_sum(output1 * target1)
                union1 = tf.reduce_sum(output1 * output1) + tf.reduce_sum(target1 * target1)
                dice += (2. * intersection1 + smooth) / (union1 + smooth)

            dice /= (self.nb_classes - 1)

            return tf.clip_by_value(1. - dice, 0., 1. - epsilon)

        return dice_loss

    def create(self):
        """
        Create model and return it

        :return: keras model
        """

        input_layer = Input(shape=self.input_shape)
        x = input_layer

        init_size = max(self.input_shape[:-1])
        size = init_size

        convolutions = self.convolutions
        connection = []
        i = 0

        if self.input_pyramid:
            scaled_input = []
            scaled_input.append(x)
            for i, nbc in enumerate(self.convolutions[:-1]):
                ds_input = AveragePooling3D(pool_size=(2, 2, 2))(scaled_input[i])
                scaled_input.append(ds_input)

        for i, nbc in enumerate(self.convolutions[:-1]):
            if not self.input_pyramid or (i == 0):
                x, x_before_ds = encoder_block(x, nbc, use_bn=self.encoder_use_bn,
                                               spatial_dropout=self.encoder_spatial_dropout)
            else:
                x, x_before_ds = encoder_block_pyramid(x, scaled_input[i], nbc, use_bn=self.encoder_use_bn,
                                                       spatial_dropout=self.encoder_spatial_dropout)
            connection.insert(0, x_before_ds)  # Append in reverse order for easier use in the next block

        x = convolution_block(x, self.convolutions[-1], self.encoder_use_bn, self.encoder_spatial_dropout)
        connection.insert(0, x)

        inverse_conv = self.convolutions[::-1]
        inverse_conv = inverse_conv[1:]
        decoded_layers = []

        for i, nbc in enumerate(inverse_conv):
            x = decoder_block(x, connection[i+1], nbc, use_bn=self.decoder_use_bn,
                              spatial_dropout=self.decoder_spatial_dropout)
            decoded_layers.append(x)

        if not self.deep_supervision:
            # Final activation layer
            x = Convolution3D(self.nb_classes, 1, activation='softmax')(x)
        else:
            recons_list = []
            for i, lay in enumerate(decoded_layers):
                x = Convolution3D(self.nb_classes, 1, activation='softmax')(lay)
                recons_list.append(x)
            x = recons_list[::-1]

        return Model(inputs=input_layer, outputs=x)
