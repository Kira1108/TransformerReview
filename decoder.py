import tensorflow as tf
from positional_encoding import positional_encoding
from decoder_layer import DecoderLayer


class Decoder(tf.keras.layers.Layer):
    """Desc:
        Decoder receives inputs of size (N, T) as decoder layer input, learns an embdding of size (N, T, D)
        Decoder also receives encoder output of size (N, T, D) as input, and performs attention over encoder output
        positioanl encoded the decoder input embddings(add position information)
        perform dropout on the positional encoded input embddings
        pass through serveral decoder layers[[This is the most important functionality of the decoder]]

        Parameters:
        -------------
        num_decoder_layers: int, number of decoder layers to stack over, L - layers Andrew N.G. said.
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

    def __init__(self, num_decoder_layers, input_vocab_size, max_tokens, d_model, num_heads, rate, dff, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_tokens, d_model)

        self.decoder_layers = [DecoderLayer(num_heads, d_model, dff, rate) for _ in range(num_decoder_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_out, training, look_ahead_mask, padding_mask):

        attention_weights = {}

        seq_len = tf.shape(x)[1]

        x = self.embedding(x)

        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        x += self.pos_encoding[:,:seq_len,:]

        x = self.dropout(x, training = training)

        for i, l in enumerate(self.decoder_layers):
            x, block1_attention, block2_attention = l(x, enc_out, training, look_ahead_mask, padding_mask)
            attention_weights[f"decoder_layer{i+1}_block1"] = block1_attention
            attention_weights[f"decoder_layer{i+1}_block2"] = block2_attention

        return x, attention_weights



def test_decoder():

    N = 64
    L = 2
    V = 10
    T = 32
    D = 512
    heads = 8
    dff = 2048

    decoder = Decoder(num_decoder_layers=L, input_vocab_size=V, max_tokens=T, d_model=D, num_heads=heads, rate=0.1, dff=dff)
    x = tf.random.uniform((N, T))
    enc_out = tf.random.uniform((N, T, D))
    training = True
    output, _ = decoder(x, enc_out, training, look_ahead_mask = None, padding_mask = None)
    assert output.shape == (N, T, D)
    print("pass decoder test")

if __name__ == "__main__":
    test_decoder()
