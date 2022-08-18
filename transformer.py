import tensorflow as tf
from encoder import Encoder
from decoder import Decoder
from masks import create_padding_mask, create_look_ahead_mask

class Transformer(tf.keras.models.Model):

    def __init__(self, num_layers, input_vocab_size, target_vocab_size, d_model, max_tokens, num_heads, dff, rate=0.1, **kwargs):
        super(Transformer, self).__init__(**kwargs)

        self.encoder = Encoder(
            num_encoder_layers=num_layers,
            input_vocab_size= input_vocab_size, 
            d_model=d_model, 
            max_tokens = max_tokens,
            num_heads=num_heads, 
            dff=dff, 
            rate=rate)

        self.decoder = Decoder(
            num_decoder_layers=num_layers,
            input_vocab_size= target_vocab_size,
            max_tokens = max_tokens,
            d_model = d_model,
            num_heads = num_heads,
            rate = rate,
            dff = dff
        )

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def create_masks(self, inp, tar):
        padding_mask = create_padding_mask(inp)

        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])

        decoder_padding_mask = create_padding_mask(tar)

        combined_mask = tf.maximum(decoder_padding_mask, look_ahead_mask)

        return padding_mask, combined_mask


    def call(self, inputs, training):
        inp, tar = inputs
        padding_mask, look_ahead_mask = self.create_masks(inp, tar)
        enc_output = self.encoder(inp, training, padding_mask)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights


def test_transformer():
    N = 64
    L = 2
    V = 10
    T = 32
    D = 512
    heads = 8
    dff = 2048
    rate = 0.1

    transformer = Transformer(num_layers=L, 
    input_vocab_size=V, 
    target_vocab_size=V, 
    d_model=D, 
    max_tokens=T, 
    num_heads=heads, 
    dff=dff, 
    rate=rate)


    enc_input = tf.random.uniform((N, T))
    dec_input = tf.random.uniform((N, T))

    output, _ = transformer(inputs=[enc_input, dec_input], training=False)
    assert output.shape == (N, T, V)
    print("passed transformer test.")


if __name__ == "__main__":
    test_transformer()
