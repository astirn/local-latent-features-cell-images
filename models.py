from test_mode import test_mode

# real run?
if not test_mode():
    import matplotlib as mpl
    mpl.use('Agg')

import os
import time
import pickle
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from utils import get_batches, plot


class AutoEncoder(object):

    def __init__(self, input_dim, latent_dim, conv_layers, full_layers, px_z, lr, batch_size, n_epochs, save_dir=None):

        # reset graphs
        tf.reset_default_graph()

        # save the shape of the data
        self.input_dim = input_dim

        # save latent dimensions
        self.latent_dim = latent_dim

        # save output type
        assert px_z == 'Gaussian' or px_z == 'Bernoulli'
        self.px_z = px_z

        # declare placeholders
        self.x_ph = tf.placeholder(tf.float32, [None] + self.input_dim)
        self.training = tf.placeholder(tf.bool)
        self.dropout_ph = tf.placeholder(tf.float32)

        # define initialization routines
        self.kernel_init = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32, uniform=False)
        self.weight_init = tf.contrib.layers.xavier_initializer(dtype=tf.float32, uniform=False)
        self.bias_init = tf.constant_initializer(0.0)

        # regularization
        self.dropout_prob = 0.2

        # encoder architecture
        self.encoder_conv_layers = conv_layers
        self.encoder_full_layers = full_layers

        # initialize encoder layer dimensions (decoder reverses the process)
        self.encoder_conv_shapes = [None for _ in range(len(self.encoder_conv_layers))]
        self.encoder_flat_shapes = None
        self.encoder_full_shapes = [None for _ in range(len(self.encoder_full_layers))]

        # build encoder network
        self.z_mu = self.encoder()

        # build decoder network
        self.x_hat = self.decoder()

        # build loss operation
        self.loss_op, self.mse_op = self.loss_operation()

        # declare scaling results
        self.chan_min = None
        self.chan_max = None
        self.chan_avg = None
        self.chan_std = None

        # configure training operation
        self.batch_size = batch_size
        self.learning_rate = lr
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.num_epochs = n_epochs
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = tf.contrib.layers.optimize_loss(loss=self.loss_op,
                                                        global_step=self.global_step,
                                                        learning_rate=self.learning_rate,
                                                        optimizer=self.optimizer,
                                                        summaries=['loss', 'gradients'])

        # configure logging
        self.saves_per_batch = 5
        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=1)
        self.save_dir = save_dir

    def encoder(self):

        # load x
        x = self.x_ph

        # loop over the number of convolution layers
        for i in range(len(self.encoder_conv_layers)):

            # run cnn layers
            input_dim = x.get_shape().as_list()[1:]
            x = tf.layers.conv2d(inputs=x,
                                 filters=self.encoder_conv_layers[i]['out_chan'],
                                 kernel_size=self.encoder_conv_layers[i]['k_size'],
                                 strides=[1, 1],
                                 padding='same',
                                 activation=tf.nn.elu,
                                 use_bias=True,
                                 kernel_initializer=self.kernel_init,
                                 bias_initializer=self.bias_init,
                                 name='EncConv{:d}'.format(i + 1))

            # run max pooling layers
            x = tf.layers.max_pooling2d(inputs=x,
                                        pool_size=3,
                                        strides=2,
                                        padding='same',
                                        name='EncPool{:d}'.format(i + 1))

            # apply drop out
            x = tf.layers.dropout(x, self.dropout_ph)

            # print size information
            self.encoder_conv_shapes[i] = x.get_shape().as_list()[1:]
            print('EncConv{:d}: '.format(i + 1) + str(input_dim) + ' --> ' + str(self.encoder_conv_shapes[i]))

        # flatten features to vector
        x = tf.contrib.layers.flatten(x)
        self.encoder_flat_shapes = x.get_shape().as_list()[1:][0]

        # loop over the shared fully connected layers
        for i in range(len(self.encoder_full_layers)):

            # run fully connected layers
            input_dim = x.get_shape().as_list()[1:][0]
            x = tf.layers.dense(inputs=x,
                                units=self.encoder_full_layers[i],
                                activation=tf.nn.elu,
                                use_bias=True,
                                kernel_initializer=self.weight_init,
                                bias_initializer=self.bias_init,
                                name='EncFull{:d}'.format(i + 1))

            # apply drop out
            x = tf.layers.dropout(x, self.dropout_ph)

            # print size information
            self.encoder_full_shapes[i] = x.get_shape().as_list()[1:][0]
            print('EncFull{:d}: '.format(i + 1) + str(input_dim) + ' --> ' + str(self.encoder_full_shapes[i]))

        # use name scope for better visualization
        with tf.variable_scope('EncOut') as scope:

            # run affine layer to compute latent space
            input_dim = x.get_shape().as_list()[1:][0]
            z = tf.layers.dense(inputs=x,
                                units=self.latent_dim,
                                activation=None,
                                use_bias=True,
                                kernel_initializer=self.weight_init,
                                bias_initializer=self.bias_init,
                                name='EncOut')

            # print size information
            print('EncOut: ' + str(input_dim) + ' --> ' + str(z.get_shape().as_list()[1:][0]))

        # return latent variables
        return z

    def decoder(self):

        # use latent space directly
        x = self.z_mu

        # loop over fully connected layers
        decoder_full_layers = self.encoder_full_shapes[::-1] + [self.encoder_flat_shapes]
        for i in range(len(decoder_full_layers)):

            # run fully connected layers
            input_dim = x.get_shape().as_list()[1:][0]
            x = tf.layers.dense(inputs=x,
                                units=decoder_full_layers[i],
                                activation=tf.nn.elu,
                                use_bias=True,
                                kernel_initializer=self.weight_init,
                                bias_initializer=self.bias_init,
                                name='DecFull{:d}'.format(i + 1))

            # apply drop out
            x = tf.layers.dropout(x, self.dropout_ph)

            # print size information
            print('DecFull{:d}: '.format(i + 1) + str(input_dim) + ' --> ' + str(x.get_shape().as_list()[1:][0]))

        # reshape for convolution transpose layers
        x = tf.reshape(x, shape=([-1] + self.encoder_conv_shapes[-1]))

        # loop over the convolution layers
        decoder_conv_layers = self.encoder_conv_shapes[:-1][::-1] + [self.input_dim]
        for i in range(len(decoder_conv_layers) - 1):

            # run convolution transpose layers
            input_dim = x.get_shape().as_list()[1:]
            x = tf.layers.Conv2DTranspose(filters=decoder_conv_layers[i][-1],
                                          kernel_size=decoder_conv_layers[i][:2],
                                          strides=[1, 1],
                                          padding='SAME',
                                          activation=tf.nn.elu,
                                          use_bias=True,
                                          kernel_initializer=self.kernel_init,
                                          bias_initializer=self.bias_init,
                                          name='DecConv{:d}'.format(i + 1))(x)

            # up-sample data
            size = [int(x.shape[1] * 2), int(x.shape[2] * 2)]
            x = tf.image.resize_bilinear(x, size=size, align_corners=None, name=None)

            # apply drop out
            x = tf.layers.dropout(x, self.dropout_ph)

            # print size information
            print('DecConv{:d}: '.format(i + 1) + str(input_dim) + ' --> ' + str(x.get_shape().as_list()[1:]))

        # use name scope for better visualization
        with tf.variable_scope('DecOut') as scope:

            # select activation type
            if self.px_z == 'Gaussian':

                # run final convolution transpose layer
                input_dim = x.get_shape().as_list()[1:]
                x = tf.layers.Conv2DTranspose(filters=decoder_conv_layers[-1][-1],
                                              kernel_size=decoder_conv_layers[-1][:2],
                                              strides=[1, 1],
                                              padding='SAME',
                                              activation=tf.nn.elu,
                                              use_bias=True,
                                              kernel_initializer=self.kernel_init,
                                              bias_initializer=self.bias_init,
                                              name='xHat')(x)

                # adjust for elu on (-1, inf)
                x = x + tf.constant(1.0, dtype=tf.float32)

                # up-sample data
                size = [int(x.shape[1] * 2), int(x.shape[2] * 2)]
                x = tf.image.resize_bilinear(x, size=size, align_corners=None, name=None)

            elif self.px_z == 'Bernoulli':

                # run final convolution transpose layer
                input_dim = x.get_shape().as_list()[1:]
                x = tf.layers.Conv2DTranspose(filters=decoder_conv_layers[-1][-1],
                                              kernel_size=decoder_conv_layers[-1][:2],
                                              strides=[1, 1],
                                              padding='SAME',
                                              activation=tf.nn.sigmoid,
                                              use_bias=True,
                                              kernel_initializer=self.kernel_init,
                                              bias_initializer=self.bias_init,
                                              name='xHat')(x)

                # up-sample data
                size = [int(x.shape[1] * 2), int(x.shape[2] * 2)]
                x = tf.image.resize_bilinear(x, size=size, align_corners=None, name=None)

            # error case
            else:
                input_dim = None
                x = None

            # print size information
            assert self.input_dim == x.get_shape().as_list()[1:]
            print('DecOut: ' + str(input_dim) + ' --> ' + str(self.input_dim))

        return x

    def loss_operation(self):

        # use name scope for better visualization
        with tf.variable_scope('Loss') as scope:

            # flatten input and reconstruction
            x = tf.layers.flatten(self.x_ph)
            x_hat = tf.layers.flatten(self.x_hat)

            # compute mse
            sq_diff = tf.squared_difference(x, x_hat)
            mse = tf.reduce_mean(tf.reduce_mean(sq_diff, axis=1))

            # Gaussian output treatment
            if self.px_z == 'Gaussian':

                # compute MSE reconstruction loss if
                loss = mse

            # Bernoulli output treatment
            elif self.px_z == 'Bernoulli':

                # compute logistic loss
                loss = -tf.reduce_sum(x * tf.log(x_hat + 1e-6) + (1 - x) * tf.log(1 - x_hat + 1e-6), axis=1)
                loss = tf.reduce_mean(loss)

        return loss, mse

    def feed_dict_samples(self, x_batch, training):

        # initialize feed dictionary
        feed_dict = dict()

        # load feed dictionary with data
        feed_dict.update({self.x_ph: x_batch})

        # load feed dictionary with training hyper-parameters
        feed_dict.update({self.training: training})
        feed_dict.update({self.dropout_ph: self.dropout_prob * float(training)})

        return feed_dict


class VariationalAutoEncoder(object):

    def __init__(self, input_dim, latent_dim, conv_layers, full_layers, px_z, lr, batch_size, n_epochs, full_var=False, save_dir=None):

        # reset graphs
        tf.reset_default_graph()

        # save the shape of the data
        self.input_dim = input_dim
        self.n_inputs = np.prod(self.input_dim)

        # save latent dimensions
        self.latent_dim = latent_dim

        # save output type
        assert px_z == 'Gaussian' or px_z == 'Bernoulli'
        self.px_z = px_z

        # declare placeholders
        self.x_ph = tf.placeholder(tf.float32, [None] + self.input_dim)
        self.z_ph = tf.placeholder(tf.float32, [None, self.latent_dim])
        self.training = tf.placeholder(tf.bool)
        self.dropout_ph = tf.placeholder(tf.float32)

        # define initialization routines
        self.kernel_init = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32, uniform=False)
        self.weight_init = tf.contrib.layers.xavier_initializer(dtype=tf.float32, uniform=False)
        self.bias_init = tf.constant_initializer(0.0)

        # regularization
        self.std_min_value = 0.1
        self.full_var = full_var
        self.dropout_prob = 0.2

        # encoder architecture
        self.encoder_conv_layers = conv_layers
        self.encoder_full_layers = full_layers

        # initialize encoder layer dimensions (decoder reverses the process)
        self.encoder_conv_shapes = [None for _ in range(len(self.encoder_conv_layers))]
        self.encoder_flat_shapes = None
        self.encoder_full_shapes = [None for _ in range(len(self.encoder_full_layers))]

        # build encoder network
        self.z_mu, self.z_std = self.encoder()

        # build decoder network
        self.x_hat, self.x_hat_var = self.decoder()

        # build loss operation
        self.loss_op, self.rec_loss_op, self.lat_loss_op, self.mse_op = self.loss_operation()

        # declare scaling results
        self.chan_min = None
        self.chan_max = None
        self.chan_avg = None
        self.chan_std = None

        # configure training operation
        self.batch_size = batch_size
        self.learning_rate = lr
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.num_epochs = n_epochs
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = tf.contrib.layers.optimize_loss(loss=self.loss_op,
                                                        global_step=self.global_step,
                                                        learning_rate=self.learning_rate / self.n_inputs,
                                                        optimizer=self.optimizer,
                                                        summaries=['loss', 'gradients'])

        # configure logging
        self.saves_per_batch = 5
        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=1)
        self.save_dir = save_dir

    def encoder(self):

        # load x
        x = self.x_ph

        # loop over the number of convolution layers
        for i in range(len(self.encoder_conv_layers)):

            # run cnn layers
            input_dim = x.get_shape().as_list()[1:]
            x = tf.layers.conv2d(inputs=x,
                                 filters=self.encoder_conv_layers[i]['out_chan'],
                                 kernel_size=self.encoder_conv_layers[i]['k_size'],
                                 strides=[1, 1],
                                 padding='same',
                                 activation=tf.nn.elu,
                                 use_bias=True,
                                 kernel_initializer=self.kernel_init,
                                 bias_initializer=self.bias_init,
                                 name='EncConv{:d}'.format(i + 1))

            # run max pooling layers
            x = tf.layers.max_pooling2d(inputs=x,
                                        pool_size=3,
                                        strides=2,
                                        padding='same',
                                        name='EncPool{:d}'.format(i + 1))

            # apply drop out
            x = tf.layers.dropout(x, self.dropout_ph)

            # print size information
            self.encoder_conv_shapes[i] = x.get_shape().as_list()[1:]
            print('EncConv{:d}: '.format(i + 1) + str(input_dim) + ' --> ' + str(self.encoder_conv_shapes[i]))

        # flatten features to vector
        x = tf.contrib.layers.flatten(x)
        self.encoder_flat_shapes = x.get_shape().as_list()[1:][0]

        # loop over the shared fully connected layers
        for i in range(len(self.encoder_full_layers)):

            # run fully connected layers
            input_dim = x.get_shape().as_list()[1:][0]
            x = tf.layers.dense(inputs=x,
                                units=self.encoder_full_layers[i],
                                activation=tf.nn.elu,
                                use_bias=True,
                                kernel_initializer=self.weight_init,
                                bias_initializer=self.bias_init,
                                name='EncFull{:d}'.format(i + 1))

            # apply drop out
            x = tf.layers.dropout(x, self.dropout_ph)

            # print size information
            self.encoder_full_shapes[i] = x.get_shape().as_list()[1:][0]
            print('EncFull{:d}: '.format(i + 1) + str(input_dim) + ' --> ' + str(self.encoder_full_shapes[i]))

        # use name scope for better visualization
        with tf.variable_scope('EncOut') as scope:

            # get input dimensions
            input_dim = x.get_shape().as_list()[1:][0]

            # compute z mean
            z_mu = tf.layers.dense(inputs=x,
                                   units=self.latent_dim,
                                   activation=None,
                                   use_bias=True,
                                   kernel_initializer=self.weight_init,
                                   bias_initializer=self.bias_init,
                                   name='zMean')

            # compute z std
            z_std = tf.layers.dense(inputs=x,
                                    units=self.latent_dim,
                                    activation=tf.nn.elu,
                                    use_bias=True,
                                    kernel_initializer=self.weight_init,
                                    bias_initializer=self.bias_init,
                                    name='zSTD')

            # adjust for elu on (-1, inf) and add minimum value
            z_std = z_std + tf.constant(1.0, dtype=tf.float32) + self.std_min_value

            # print size information
            assert z_mu.get_shape().as_list()[1:][0] == z_std.get_shape().as_list()[1:][0]
            print('EncOut: ' + str(input_dim) + ' --> ' + str(z_mu.get_shape().as_list()[1:][0]))

        # return latent variables
        return z_mu, z_std

    def decoder(self):

        # use name scope for better visualization
        with tf.variable_scope('ReparameterizationTrick') as scope:

            # apply sampling (shift and scale)
            x = self.z_mu + self.z_ph * self.z_std

        # loop over fully connected layers
        decoder_full_layers = self.encoder_full_shapes[::-1] + [self.encoder_flat_shapes]
        for i in range(len(decoder_full_layers)):

            # run fully connected layers
            input_dim = x.get_shape().as_list()[1:][0]
            x = tf.layers.dense(inputs=x,
                                units=decoder_full_layers[i],
                                activation=tf.nn.elu,
                                use_bias=True,
                                kernel_initializer=self.weight_init,
                                bias_initializer=self.bias_init,
                                name='DecFull{:d}'.format(i + 1))

            # apply drop out
            x = tf.layers.dropout(x, self.dropout_ph)

            # print size information
            print('DecFull{:d}: '.format(i + 1) + str(input_dim) + ' --> ' + str(x.get_shape().as_list()[1:][0]))

        # reshape for convolution transpose layers
        x = tf.reshape(x, shape=([-1] + self.encoder_conv_shapes[-1]))

        # loop over the convolution layers
        decoder_conv_layers = self.encoder_conv_shapes[:-1][::-1] + [self.input_dim]
        for i in range(len(decoder_conv_layers) - 1):

            # run convolution transpose layers
            input_dim = x.get_shape().as_list()[1:]
            x = tf.layers.Conv2DTranspose(filters=decoder_conv_layers[i][-1],
                                          kernel_size=decoder_conv_layers[i][:2],
                                          strides=[1, 1],
                                          padding='SAME',
                                          activation=tf.nn.elu,
                                          use_bias=True,
                                          kernel_initializer=self.kernel_init,
                                          bias_initializer=self.bias_init,
                                          name='DecConv{:d}'.format(i + 1))(x)

            # up-sample data
            size = [int(x.shape[1] * 2), int(x.shape[2] * 2)]
            x = tf.image.resize_bilinear(x, size=size, align_corners=None, name=None)

            # apply drop out
            x = tf.layers.dropout(x, self.dropout_ph)

            # print size information
            print('DecConv{:d}: '.format(i + 1) + str(input_dim) + ' --> ' + str(x.get_shape().as_list()[1:]))

        # use name scope for better visualization
        with tf.variable_scope('DecOut') as scope:

            # select activation type
            if self.px_z == 'Gaussian':

                # run final convolution transpose layer
                input_dim = x.get_shape().as_list()[1:]
                x_hat_mu = tf.layers.Conv2DTranspose(filters=decoder_conv_layers[-1][-1],
                                                     kernel_size=decoder_conv_layers[-1][:2],
                                                     strides=[1, 1],
                                                     padding='SAME',
                                                     activation=tf.nn.elu,
                                                     use_bias=True,
                                                     kernel_initializer=self.kernel_init,
                                                     bias_initializer=self.bias_init,
                                                     name='xHatMu'.format(len(decoder_conv_layers)))(x)

                # adjust for elu on (-1, inf)
                x_hat_mu = x_hat_mu + tf.constant(1.0, dtype=tf.float32)

                # up-sample data
                size = [int(x_hat_mu.shape[1] * 2), int(x_hat_mu.shape[2] * 2)]
                x_hat_mu = tf.image.resize_bilinear(x_hat_mu, size=size, align_corners=None, name=None)

                # compute standard deviation
                x_hat_std = tf.layers.Conv2DTranspose(filters=decoder_conv_layers[-1][-1],
                                                      kernel_size=decoder_conv_layers[-1][:2],
                                                      strides=[1, 1],
                                                      padding='SAME',
                                                      activation=tf.nn.elu,
                                                      use_bias=True,
                                                      kernel_initializer=self.kernel_init,
                                                      bias_initializer=self.bias_init,
                                                      name='xHatStd'.format(len(decoder_conv_layers)))(x)
                # up-sample data
                size = [int(x_hat_std.shape[1] * 2), int(x_hat_std.shape[2] * 2)]
                x_hat_std = tf.image.resize_bilinear(x_hat_std, size=size, align_corners=None, name=None)

                if not self.full_var:

                    # flatten features to vector
                    x_hat_std = tf.contrib.layers.flatten(x_hat_std)

                    # compute z std
                    x_hat_std = tf.layers.dense(inputs=x_hat_std,
                                                units=1,
                                                activation=tf.nn.elu,
                                                use_bias=True,
                                                kernel_initializer=self.weight_init,
                                                bias_initializer=self.bias_init,
                                                name='xHatStdScalar')

                # adjust for elu on (-1, inf) and add minimum value
                x_hat_std = x_hat_std + tf.constant(1.0, dtype=tf.float32) + self.std_min_value

                # convert to variance and flatten
                x_hat_var = tf.layers.flatten(x_hat_std ** 2)

            elif self.px_z == 'Bernoulli':

                # run final convolution transpose layer
                input_dim = x.get_shape().as_list()[1:]
                x_hat_mu = tf.layers.Conv2DTranspose(filters=decoder_conv_layers[-1][-1],
                                                     kernel_size=decoder_conv_layers[-1][:2],
                                                     strides=[1, 1],
                                                     padding='SAME',
                                                     activation=tf.nn.sigmoid,
                                                     use_bias=True,
                                                     kernel_initializer=self.kernel_init,
                                                     bias_initializer=self.bias_init,
                                                     name='DecConv{:d}'.format(len(decoder_conv_layers)))(x)

                # up-sample data
                size = [int(x_hat_mu.shape[1] * 2), int(x_hat_mu.shape[2] * 2)]
                x_hat_mu = tf.image.resize_bilinear(x_hat_mu, size=size, align_corners=None, name=None)

                # no variance for this model
                x_hat_var = None

            # error case
            else:
                input_dim = None
                x_hat_mu = None
                x_hat_var = None

            # print size information
            assert self.input_dim == x_hat_mu.get_shape().as_list()[1:]
            print('DecOut: ' + str(input_dim) + ' --> ' + str(self.input_dim))

        return x_hat_mu, x_hat_var

    def loss_operation(self):

        # use name scope for better visualization
        with tf.variable_scope('Loss') as scope:

            # flatten input and reconstruction
            x = tf.layers.flatten(self.x_ph)
            x_hat_mu = tf.layers.flatten(self.x_hat)

            # compute mse
            sq_diff = tf.squared_difference(x, x_hat_mu)
            mse = tf.reduce_mean(tf.reduce_mean(sq_diff, axis=1))

            # Gaussian output treatment
            if self.px_z == 'Gaussian':

                # compute reconstruction loss: -E[p(x|z)]
                if self.full_var:
                    log_det = -0.5 * tf.reduce_sum(tf.log(2 * np.pi * self.x_hat_var), axis=1)
                else:
                    log_det = -0.5 * tf.log(2 * np.pi * self.x_hat_var) * self.n_inputs
                log_exp = -0.5 * tf.reduce_sum(sq_diff / self.x_hat_var, axis=1)
                rec_loss = -tf.reduce_mean(log_exp + log_det)

            # Bernoulli output treatment
            elif self.px_z == 'Bernoulli':

                # compute logistic loss
                rec_loss = -tf.reduce_sum(x * tf.log(x_hat_mu + 1e-6) + (1 - x) * tf.log(1 - x_hat_mu + 1e-6), axis=1)
                rec_loss = tf.reduce_mean(rec_loss)

            # error case
            else:
                rec_loss = None

            # convert to tensors of shape [None, Latent Space Dimensions]
            z_mean = self.z_mu
            z_std = self.z_std

            # compute latent loss per sample: KL(q(z) || p(z))
            lat_loss = -0.5 * tf.reduce_sum(1 + tf.log(tf.square(z_std)) - tf.square(z_mean) - tf.square(z_std), axis=1)
            lat_loss = tf.reduce_mean(lat_loss)

            # compute total loss: -(E[p(x|z)] - KL(q(z) || p(z))) = -E[p(x|z)] + KL(q(z) || p(z)))
            loss = rec_loss + lat_loss

        return loss, rec_loss, lat_loss, mse

    def feed_dict_samples(self, x_batch, training):

        # initialize feed dictionary
        feed_dict = dict()

        # get batch size
        batch_size =x_batch.shape[0]

        # take center and width samples from standard normal: N(0,I)
        z_samps = np.random.normal(size=[batch_size, self.latent_dim])

        # load feed dictionary with data
        feed_dict.update({self.x_ph: x_batch})
        feed_dict.update({self.z_ph: z_samps})

        # load feed dictionary with training hyper-parameters
        feed_dict.update({self.training: training})
        feed_dict.update({self.dropout_ph: self.dropout_prob * float(training)})

        return feed_dict


def train(mdl, x_train, x_test):

    # running Gaussian loss
    if mdl.px_z == 'Gaussian':

        # scale each channel to [0, 100]
        chan_max = np.zeros(x_train.shape[-1])
        chan_min = np.zeros(x_train.shape[-1])
        for i in range(x_train.shape[-1]):
            chan_max[i] = np.max([np.max(x_train[:, :, :, i]), np.max(x_test[:, :, :, i])])
            chan_min[i] = np.min([np.min(x_train[:, :, :, i]), np.min(x_test[:, :, :, i])])
            x_train[:, :, :, i] = 100 * (x_train[:, :, :, i] - chan_min[i]) / (chan_max[i] - chan_min[i])
            x_test[:, :, :, i] = 100 * (x_test[:, :, :, i] - chan_min[i]) / (chan_max[i] - chan_min[i])

        assert np.min(x_train) >= 0 and np.min(x_test) >= 0
        assert np.max(x_train) <= 101 and np.max(x_test) <= 101

        # save scaling results
        mdl.chan_min = chan_min
        mdl.chan_max = chan_max

    # running Bernoulli loss
    elif mdl.px_z == 'Bernoulli':

        # whiten each channel
        chan_avg = np.zeros(x_train.shape[-1])
        chan_std = np.zeros(x_train.shape[-1])
        for i in range(x_train.shape[-1]):
            chan_avg[i] = np.mean(x_train[:, :, :, i])
            chan_std[i] = np.std(x_train[:, :, :, i])
            x_train[:, :, :, i] = (x_train[:, :, :, i] - chan_avg[i]) / chan_std[i]
            x_test[:, :, :, i] = (x_test[:, :, :, i] - chan_avg[i]) / chan_std[i]

        # convert to {0, 1}
        x_train = np.sign(x_train)
        x_train[x_train <= 0] = 0
        x_test = np.sign(x_test)
        x_test[x_test <= 0] = 0
        assert np.min(x_train) == np.min(x_test) == 0
        assert np.max(x_train) == np.max(x_test) == 1

        # save scaling results
        mdl.chan_avg = chan_avg
        mdl.chan_std = chan_std

    # pickle the model parameters
    if not os.path.exists(mdl.save_dir):
        os.makedirs(mdl.save_dir)
    mdl_params = {'input_dim': mdl.input_dim,
                  'latent_dim': mdl.latent_dim,
                  'conv_layers': mdl.encoder_conv_layers,
                  'full_layers': mdl.encoder_full_layers,
                  'lr': mdl.learning_rate,
                  'px_z': mdl.px_z,
                  'batch_size': mdl.batch_size,
                  'n_epochs': mdl.num_epochs,
                  'save_dir': mdl.save_dir,
                  'chan_min': mdl.chan_min,
                  'chan_max': mdl.chan_max,
                  'chan_avg': mdl.chan_avg,
                  'chan_std': mdl.chan_std}
    pickle.dump(mdl_params, open(os.path.join(mdl.save_dir, 'mdl_params.p'), 'wb'))

    # declare figure handles for plots updated during training
    fig_image = plt.figure()
    fig_loss = plt.figure()

    # start session
    with tf.Session() as sess:

        # run initialization
        sess.run(tf.global_variables_initializer())

        # configure writer if save directory is specified
        if mdl.save_dir is not None:
            train_writer = tf.summary.FileWriter(mdl.save_dir, sess.graph)
        else:
            train_writer = None

        # loop over the number of epochs
        train_loss_hist = []
        train_mse_hist = []
        test_loss_hist = np.zeros(mdl.num_epochs)
        test_mse_hist = np.zeros(mdl.num_epochs)
        for i in range(mdl.num_epochs):

            # start timer
            start = time.time()

            # get training batches
            batches = get_batches(x_train.shape[0], mdl.batch_size)

            # configure writer to write five times for this epoch
            write_interval = np.ceil(len(batches) / mdl.saves_per_batch)

            # loop over the batches
            for j in range(len(batches)):

                # load a feed dictionary
                feed_dict = mdl.feed_dict_samples(x_train[batches[j]], True)

                # writing to tensor board?
                if np.mod(j, write_interval) == 0 and train_writer is not None:

                    # run training, loss, accuracy, and summary
                    _, loss, mse, summary = sess.run([mdl.train_op,
                                                      mdl.loss_op,
                                                      mdl.mse_op,
                                                      mdl.merged], feed_dict=feed_dict)

                    # write the summary
                    train_writer.add_summary(summary, mdl.global_step.eval(sess))

                # not writing to tensor board
                else:
                    # run just training and loss
                    _, loss, mse = sess.run([mdl.train_op, mdl.loss_op, mdl.mse_op], feed_dict=feed_dict)

                # test for NaN and exit if so
                if np.isnan(loss):
                    print('\n Nan whelp!')
                    return

                # print update
                per = 100 * (j + 1) / len(batches)
                update_str = 'Epoch %d, Percent Complete = %f%%, MSE = %f' % (i + 1, per, mse)
                print('\r' + update_str, end='')

                # plot update
                if np.mod(j, write_interval) == 0:

                    # save total loss
                    train_loss_hist.append(loss)
                    train_mse_hist.append(mse)

                    # clear the figure
                    fig_loss.clf()
                    fig_loss.suptitle('Learning Curve')

                    # generate epoch numbers
                    batch_epochs = np.arange(1, len(train_loss_hist) + 1) / mdl.saves_per_batch
                    epochs = np.arange(1, i + 1)

                    # plot the loss
                    sp = fig_loss.add_subplot(1, 2, 1)
                    plt.plot(batch_epochs, train_loss_hist, label='train batch loss')
                    plt.plot(epochs, test_loss_hist[:i], label='test loss')
                    plt.legend()
                    sp.set_xlabel('Epoch')
                    sp.set_ylabel('Loss')

                    # plot the MSE
                    sp = fig_loss.add_subplot(1, 2, 2)
                    plt.plot(batch_epochs, train_mse_hist, label='train batch mse')
                    plt.plot(epochs, test_mse_hist[:i], label='test mse')
                    plt.legend()
                    sp.set_xlabel('Epoch')
                    sp.set_ylabel('Batch MSE')
                    if mdl.save_dir is None:
                        plt.pause(0.05)
                    else:
                        fig_loss.savefig(os.path.join(mdl.save_dir, 'Learning_Curve'))

            # print time for epoch
            stop = time.time()
            print('\nTime for Epoch = %f' % (stop - start))

            # get test loss
            test_loss_hist[i], test_mse_hist[i] = test_loss(mdl, sess, x_test)

            # is the current test performance the best?
            if test_mse_hist[i] == np.min(test_mse_hist[:i + 1]) and mdl.save_dir is not None:

                # save model
                save_path = mdl.saver.save(sess, os.path.join(mdl.save_dir, 'model.ckpt'))
                print('New best model! Saving results to ' + save_path)

            # grab random scan
            i_plot = np.random.choice(x_train.shape[0])
            x = np.expand_dims(x_train[i_plot], axis=0)

            # generate test image
            feed_dict = mdl.feed_dict_samples(x, False)

            # run the optimizer
            x_hat = sess.run(mdl.x_hat, feed_dict=feed_dict)

            # plot image
            im_title = ('Model_Performance_Epoch_%d' % i)
            plot(x[0],
                 x_hat[0],
                 fig=fig_image,
                 super_title=im_title,
                 save_loc=os.path.join(mdl.save_dir, im_title))

        # close session
        sess.close()


def test_loss(mdl, sess, x_test):

    # get training batches
    batches = get_batches(x_test.shape[0], mdl.batch_size)
    batch_lengths = np.array([len(batch) for batch in batches])

    # initialize results
    loss = np.zeros(len(batches))
    mse = np.zeros(len(batches))

    # loop over the batches
    for j in range(len(batches)):

        # load a feed dictionary
        feed_dict = mdl.feed_dict_samples(x_test[batches[j]], False)

        # compute metrics
        loss[j], mse[j] = sess.run([mdl.loss_op, mdl.mse_op], feed_dict=feed_dict)

        # print update
        per = 100 * (j + 1) / len(batches)
        update_str = 'Evaluating test set performance. Percent Complete = {:.2f}%'.format(per)
        print('\r' + update_str, end='')

    # take the average
    loss = np.sum(loss * batch_lengths) / np.sum(batch_lengths)
    mse = np.sum(mse * batch_lengths) / np.sum(batch_lengths)

    # results
    print('\nTest Loss = {:f}, Test MSE = {:f}'.format(loss, mse))

    return loss, mse


def generate_latent_matrix(mdl, x):

    # begin new session
    with tf.Session() as sess:

        # run initialization
        sess.run(tf.global_variables_initializer())

        # load the model
        mdl.saver.restore(sess, os.path.join(mdl.save_dir, 'model.ckpt'))

        # get training batches
        batches = get_batches(x.shape[0], mdl.batch_size, shuffle=False)

        # initialize results
        z = np.zeros([x.shape[0], mdl.latent_dim])

        # loop over the batches
        for j in range(len(batches)):

            # load a feed dictionary
            feed_dict = mdl.feed_dict_samples(x[batches[j]], False)

            # compute latent space
            z[batches[j]] = sess.run(mdl.z_mu, feed_dict=feed_dict)

            # print update
            per = 100 * (j + 1) / len(batches)
            update_str = 'Computing Latent Space. Percent Complete = {:.2f}%'.format(per)
            print('\r' + update_str, end='')

        # new line
        print('')

    # save latent space
    np.save(os.path.join(mdl.save_dir, 'z.npy'), z)


def mdl_reconstruction(sess, mdl, mdl_params, x):

    # running Gaussian loss
    if mdl.px_z == 'Gaussian':

        # scale each channel to [0, 100]
        for i in range(x.shape[-1]):
            x[:, :, :, i] = 100 * (x[:, :, :, i] - mdl_params['chan_min'][i]) / (mdl_params['chan_max'][i] - mdl_params['chan_min'][i])

        assert np.min(x) >= 0
        assert np.max(x) <= 101

    # running Bernoulli loss
    elif mdl.px_z == 'Bernoulli':

        # whiten each channel
        for i in range(x.shape[-1]):
            x[:, :, :, i] = (x[:, :, :, i] - mdl_params['chan_avg'][i]) / mdl_params['chan_std'][i]

        # convert to {0, 1}
        x = np.sign(x)
        x[x <= 0] = 0
        assert np.min(x) == 0
        assert np.max(x) == 1

    # load a feed dictionary
    feed_dict = mdl.feed_dict_samples(x, False)

    # compute latent space
    x_hat = sess.run(mdl.x_hat, feed_dict=feed_dict)

    # bernoulli
    if mdl.px_z == 'Bernoulli':
        x_hat = np.sign(x_hat - 0.5)
        x_hat[x_hat == -1] = 0

    return x, x_hat


if __name__ == '__main__':

    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = np.reshape(mnist.train.images, [-1, 28, 28, 1])
    valid_data = np.reshape(mnist.test.images, [-1, 28, 28, 1])

    # build AE model
    ae = AutoEncoder(input_dim=[28, 28, 1],
                     latent_dim=8,
                     conv_layers=[{'k_size': 5, 'out_chan': 3}, {'k_size': 3, 'out_chan': 3}],
                     full_layers=[100, 75],
                     lr=1e-4,
                     px_z='Bernoulli',
                     batch_size=100,
                     n_epochs=100,
                     save_dir='Test/BernAE/')

    # train the model
    train(ae, train_data, valid_data)

    # build AE model
    ae = AutoEncoder(input_dim=[28, 28, 1],
                     latent_dim=8,
                     conv_layers=[{'k_size': 5, 'out_chan': 3}, {'k_size': 3, 'out_chan': 3}],
                     full_layers=[100, 75],
                     lr=1e-4,
                     px_z='Gaussian',
                     batch_size=100,
                     n_epochs=100,
                     save_dir='Test/GaussAE/')

    # train the model
    train(ae, train_data, valid_data)

    # build VAE model
    vae = VariationalAutoEncoder(input_dim=[28, 28, 1],
                                 latent_dim=8,
                                 conv_layers=[{'k_size': 5, 'out_chan': 3}, {'k_size': 3, 'out_chan': 3}],
                                 full_layers=[100, 75],
                                 lr=1e-4,
                                 px_z='Bernoulli',
                                 full_var=False,
                                 batch_size=100,
                                 n_epochs=100,
                                 save_dir='Test/BernVAE/')

    # train the model
    train(vae, train_data, valid_data)

    # build VAE model
    vae = VariationalAutoEncoder(input_dim=[28, 28, 1],
                                 latent_dim=8,
                                 conv_layers=[{'k_size': 5, 'out_chan': 3}, {'k_size': 3, 'out_chan': 3}],
                                 full_layers=[100, 75],
                                 lr=1e-4,
                                 px_z='Gaussian',
                                 full_var=False,
                                 batch_size=100,
                                 n_epochs=100,
                                 save_dir='Test/GaussVAE/')

    # train the model
    train(vae, train_data, valid_data)

    # keep plots alive
    print('All done!')
    plt.ioff()
    plt.show()
