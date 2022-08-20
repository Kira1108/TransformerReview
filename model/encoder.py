import tensorflow as tf
from .positional_encoding import positional_encoding
from .encoder_layer import EncoderLayer


class Encoder(tf.keras.layers.Layer):
    """Desc:
        Encoder receives inputs of size (N, T), learns an embdding of size (N, T, D)
        positioanl encoded the input embddings(add position information)
        perform dropout on the positional encoded input embddings
        pass through serveral encoder layers[[This is the most important functionality of the encoder]]

        Parameters:
        -------------
        num_encoder_layers: int, number of encoder layers to stack over, L - layers Andrew N.G. said.
        input_vocab_size: int, size of the input vocabulary(V)
        d_model: int, size of embdding dimension(D)
        max_tokens: int, maximum number of tokens in a sequence(T)
        num_heads: int, number of heads to use in multi-head attention
        dff: int, size of the intermediate dense layer(DFF)
        rate: float, dropout rate

        Returns:
        -------------
        output: tensor, output of the encoder, of size (N, T, D) == (N, max_tokens, d_model)
    """

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

        self.encoder_layers = [EncoderLayer(num_heads, d_model, dff, rate) for _ in range(self.num_encoder_layers)]


    def call(self, x, training, mask):
        x = self.embedding(x)

        # multiply with sqrt(d_model), I dont really know what is this.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # positional encoding, which is added to input embdddings
        x += self.pos_encoding[:, :tf.shape(x)[1], :]

        # dropout
        x = self.dropout(x, training = training)

        # stacking encoder layers
        for enc_layer in self.encoder_layers:
            x = enc_layer(x, training, mask)
        
        return x

def test_encoder():

    N = 64
    T =128
    V =256
    D = 512
    heads = 8
    dff = 2048
    layers = 2

    encoder = Encoder(
        num_encoder_layers = layers, 
        input_vocab_size = V, 
        d_model = D, 
        max_tokens = T, 
        num_heads = heads, 
        dff = dff)

    inputs = tf.random.uniform((N, T))
    output = encoder(inputs, training = False, mask = None)
    assert output.shape == (N, T, D)
    print("passed encoder test.")


if __name__ == "__main__":
    test_encoder()
