import numpy as np
import tensorflow as tf


def get_angles(pos,i,d_model):
    # i//2 convert an array [0,1,2,3,....] into [0,0,1,1,2,2,3,3,....]
    # this matches the positional encoding formula, with i as index on both odd and even index.
    return pos / np.power(10000, 2*(i//2)/np.float32(d_model))

def positional_encoding(position, d_model):
    """get positional encoding matrix
    -------
    Parameters:
        position: int, sequence length
        d_model: int, embedding dimension

    Returns:
        a 3 dimensional tensor of shape [1, position, d_model]

    Usage:
        add this 3 dimensional tensor to the input embedding
        since the first dimension of size 1, the adding will be broadcasted into
        all training samples.
    """

    angles = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis,:],
        d_model
    )

    angles[:,0::2] = np.sin(angles[:,0::2])
    angles[:,1::2] = np.sin(angles[:,1::2])

    pos_encoding = angles[np.newaxis, ...]

    return tf.cast(pos_encoding, tf.float32)