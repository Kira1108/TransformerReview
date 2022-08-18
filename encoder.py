import tensorflow as tf


class Encoder(tf.keras.layers.Layer):

    def __init__(self, input_vocab_size, d_model, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)

        # now I have to write positional encoding before the encoder


    def call(self, x, training, mask):
        x = self.embedding(x)

        # multiply with sqrt(d_model)

        # positional encoding, only require a sequence length and an embdding dimension == d_model

        # dropout

        # stacking encoder layers

        # no dense layers at last
