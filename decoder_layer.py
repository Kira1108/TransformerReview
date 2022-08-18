import tensorflow as tf
from mha import MultiHeadAttention
from feed_forward import point_wise_feed_forward_network

class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, num_heads, d_model, dff, rate = 0.1, **kwargs):
        super().__init__(**kwargs)

        self.num_heads = num_heads
        self.d_model = d_model

        self.mha1 = MultiHeadAttention(self.num_heads, self.d_model)
        self.mha2 = MultiHeadAttention(self.num_heads, self.d_model)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.feed_forward = point_wise_feed_forward_network(self.d_model, dff)


    def call(self, x, enc_out, training, padding_mask, look_ahead_mask):

        out1, attn1 = self.mha1(x, x, x, look_ahead_mask)
        out1 = self.dropout1(out1, training = training)
        out1 = self.layernorm1(out1 + x)


        out2, attn2 = self.mha2(enc_out, enc_out, out1, padding_mask)
        out2 = self.dropout2(out2, training = training)
        out2 = self.layernorm2(out2 + out1)

        out3 = self.feed_forward(out2)
        out3 = self.dropout3(out3, training = training)
        out3 = self.layernorm3(out3 + out2)

        return out3, attn1, attn2


def test_decoder_layer():
    N, T, D, heads= (32, 10, 512, 8)
    x = tf.random.uniform((N, T, D))

    # here you have to pass a mask with (N, heads, T, T) 
    # to match tensors already splited into multiple heads
    mask = tf.cast(tf.random.uniform((N, heads, T, T)) > 0.5, tf.float32)
    enc_out = tf.random.uniform((N, T, D))
    decoder_layer = DecoderLayer(heads, D, D, rate = 0.1)
    out = decoder_layer(x, enc_out, training = True, padding_mask = mask, look_ahead_mask = mask)
    assert out[0].shape == (N,T,D)
    print("decoder layer test passed")


if __name__ == "__main__":
    test_decoder_layer()