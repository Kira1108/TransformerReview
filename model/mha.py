"""Very Important mask notes:
Mask is only used in attention weights, masked positions will not be pay attention to.
If positions gain no attention, they have very limited effects to the final output.
"""


import tensorflow as tf

def scaled_dot_product(q,k,v, mask):
    """
    q: [batch_size, seq_len, d_k]
    k: [batch_size, seq_len, d_k]
    v: [batch_size, seq_len, d_v]
    """
    qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)

    qk = qk / tf.math.sqrt(dk)

    if mask is not None:
         qk += (mask * -1e9)

    attention_weights = tf.nn.softmax(qk, axis = -1)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, num_heads, d_model, **kwargs):

        """Multihead Attention Lyaer
            num_heads : number of heads to include in the model
            d_model : dimension of the model
        """
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.densor = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x,(batch_size, - 1,self.num_heads, self.depth))
        x = tf.transpose(x, perm = [0,2,1,3])
        return x

    def call(self,v,k,q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        out, attention_weights = scaled_dot_product(q,k,v, mask)

        out = tf.transpose(out, perm = [0,2,1,3])
        out = tf.reshape(out,(batch_size, -1, self.d_model))
        out = self.densor(out)
        return out, attention_weights



def test_multihead_attention():
    N, T, D, heads= (32, 10, 512, 8)
    q = tf.random.uniform((N, T, D))
    k = tf.random.uniform((N, T, D))
    v = tf.random.uniform((N, T, D))

    out, attention = MultiHeadAttention(heads, D)(q, k, v, mask = None)

    assert out.shape == (N,T,D)
    assert attention.shape == (N, heads, T,T)
    print("mha test passed")



def test_scaled_dot_product():
    N, T, D = (32, 10, 512)
    q = tf.random.uniform((N, T, D))
    k = tf.random.uniform((N, T, D))
    v = tf.random.uniform((N, T, D))

    out, attention = scaled_dot_product(q, k, v, mask = None)

    assert out.shape == (N,T,D)
    assert attention.shape == (N,T,T)
    print("scaled dot product test passed")

if __name__ == "__main__":
    test_scaled_dot_product()
    test_multihead_attention()
