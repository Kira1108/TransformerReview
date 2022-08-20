import tensorflow as tf
from .feed_forward import point_wise_feed_forward_network
from .mha import MultiHeadAttention


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, num_heads, d_model, dff, rate = 0.1,**kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.dff = dff

        self.mha = MultiHeadAttention(self.num_heads, self.d_model)
        self.ffn = point_wise_feed_forward_network(self.d_model, self.dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        out1, _ = self.mha(x,x,x,mask)
        out1 = self.dropout1(out1, training = training)
        out1 = self.layernorm1(out1 + x)

        out2 = self.ffn(out1)
        out2 = self.dropout2(out2, training = training)
        out2 = self.layernorm2(out2 + out1)

        return out2


def test_encoder_layer():
    N, T, D, heads= (32, 10, 512, 8)
    x = tf.random.uniform((N, T, D))

    # here you have to pass a mask with (N, heads, T, T) 
    # to match tensors already splited into multiple heads
    mask = tf.cast(tf.random.uniform((N, heads, T, T)) > 0.5, tf.float32)
    encoder_layer = EncoderLayer(heads, D, D, rate = 0.1)
    out = encoder_layer(x, training = True, mask = mask)
    assert out.shape == (N,T,D)
    print("encoder layer test passed")

if __name__ == "__main__":
    test_encoder_layer()
