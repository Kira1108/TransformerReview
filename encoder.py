import tensorflow as tf
from positional_encoding import positional_encoding
from encoder_layer import EncoderLayer


class Encoder(tf.keras.layers.Layer):

    def __init__(self, num_encoder_layers, input_vocab_size, d_model, max_tokens, num_heads, dff, rate = 0.1,**kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers

        # this is really V by D (as my teacher lazy programmer says.) 
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)

        # input is readlly T and D (as my teacher lazy programmer says.)
        # output would be (N, T[max_token], D[d_model])
        self.pos_encoding = positional_encoding(max_tokens, d_model)

        self.dropout = tf.keras.layers.Dropout(rate) 

        self.encoder_layers = [EncoderLayer(num_heads, d_model, dff, rate) for i in range(self.num_encoder_layers)]


    def call(self, x, training, mask):
        x = self.embedding(x)

        # multiply with sqrt(d_model), I dont really know what is this.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # positional encoding, only require a sequence length and an embdding dimension == d_model
        x += self.pos_encoding[:, :tf.shape(x)[1], :]

        # dropout
        x = self.dropout(x, training = training)

        # stacking encoder layers
        for enc_layer in self.encoder_layers:
            x = enc_layer(x, training, mask)
        
        return x
