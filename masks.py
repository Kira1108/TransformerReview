from venv import create
import tensorflow as tf

def create_padding_mask(x):
    # x is of size (N, T)
    # attention weights is going to be masked, which is of size (N, heads, Tq, Tk)
    # mask is perform over Tk dimension
    # so you need an (N, 1, 1, Tk) shaped binary tensor as result

    binary_seq = tf.cast(tf.math.equal(x, 0), tf.float32)

    return binary_seq[:, tf.newaxis, tf.newaxis, :] 

def create_look_ahead_mask(size):
    # create a square matrix
    # with values below diagonal = 0 and above = 1
    # so every future values is masked
    # since this matrix is of size (T, T)
    # adding to attention weights is guraranted to be size of (N,heads, Tq, Tk)
    return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

