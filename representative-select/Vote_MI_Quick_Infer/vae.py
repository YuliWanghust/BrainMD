# -*- coding: utf-8 -*-
import tensorflow as tf

class VAE(tf.keras.Model):
    def __init__(self, input_shape, args):
        super(VAE, self).__init__()
                
        # encoder
        self.encoder = gaussian_encoder(input_shape, args)
        
        # z sampling layer
        self.sampling_layer = Sampling()
        
        # decoder
        self.decoder = bernoulli_decoder(args, input_shape)

        #self.saver = tf.train.Saver()

    def KLD_loss(self, mu, sigma):
        loss = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.math.log(1e-8 + tf.square(sigma)) - 1, 1)
        loss = tf.reduce_mean(loss)

        return loss

    def BCE_loss(self, x, y):
        loss = tf.reduce_sum(x*tf.math.log(y) + (1-x)*tf.math.log(1-y), 1)

        return -tf.reduce_mean(loss)

    def call(self, x):
        mu, sigma = self.encoder(x)
        z = self.sampling_layer((mu, sigma))
        x_hat = self.decoder(z)
        x_hat = tf.clip_by_value(x_hat, 1e-8, 1-1e-8)

        # loss
        bce_loss = self.BCE_loss(x, x_hat)
        kld_loss = self.KLD_loss(mu, sigma)

        return x_hat, bce_loss, kld_loss

    ### Function for plotting
    # def save_model(self, sess, path='model.ckpt'):
    #     self.saver.save(sess, path)
    #     print(f"Model saved to {path}")
    #
    # def load_model(self, sess, path='model.ckpt'):
    #     self.saver.restore(sess, path)
    #     print(f"Model restored from {path}")

    def save_model(self, encoder_path='encoder_weights.h5', decoder_path='decoder_weights.h5'):
        self.encoder.save_weights(encoder_path)
        self.decoder.save_weights(decoder_path)
        print(f"Encoder weights saved to {encoder_path}")
        print(f"Decoder weights saved to {decoder_path}")

    def load_model(self, encoder_path='encoder_weights.h5', decoder_path='decoder_weights.h5'):
        self.encoder.load_weights(encoder_path)
        self.decoder.load_weights(decoder_path)
        print(f"Encoder weights loaded from {encoder_path}")
        print(f"Decoder weights loaded from {decoder_path}")

    def get_config(self):
        config = super(VAE, self).get_config()
        config.update({
            'encoder': self.encoder,
            'sampling_layer': self.sampling_layer,
            'decoder': self.decoder,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def gaussian_encoder(input_shape, args):
    # initializer
    w_init = tf.keras.initializers.glorot_normal(args.seed)
    
    # input
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # hidden layer
    x = tf.keras.layers.Dense(args.n_hidden, activation='elu', 
                              kernel_initializer=w_init)(inputs)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(args.n_hidden, activation='tanh',
                              kernel_initializer=w_init)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    # output
    mu = tf.keras.layers.Dense(args.dim_z, kernel_initializer=w_init)(x)
    sigma = tf.keras.layers.Dense(args.dim_z, kernel_initializer=w_init)(x)
    sigma = 1e-6 + tf.nn.softplus(sigma)
    
    encoder = tf.keras.Model(inputs, [mu, sigma])
    encoder.summary()
    
    return encoder

def bernoulli_decoder(args, n_output):
    # initializer
    w_init = tf.keras.initializers.glorot_normal(args.seed)
    
    # input
    inputs = tf.keras.layers.Input(shape=args.dim_z)
    
    # hidden layer
    x = tf.keras.layers.Dense(args.n_hidden, activation='tanh',
                              kernel_initializer=w_init)(inputs)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(args.n_hidden, activation='elu',
                              kernel_initializer=w_init)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    # output
    x = tf.keras.layers.Dense(n_output, activation='sigmoid',
                              kernel_initializer=w_init)(x)
    
    decoder = tf.keras.Model(inputs, x)
    decoder.summary()
    
    return decoder

class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mu, sigma = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]

        z = mu + sigma * tf.random.normal((batch, dim), 0, 1)

        return z