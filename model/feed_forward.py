import tensorflow as tf


def point_wise_feed_forward_network(d_model, dff):
    """d_model is output layer shape, dff is first layer shape"""
    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])


def test_point_wise_feed_forward_network():
    model = point_wise_feed_forward_network(512, 2048)
    tensor = tf.random.uniform((64, 50, 123))
    output = model(tensor)
    assert output.shape == (64, 50, 512)
    print("pointwise feedforward test passed")


if __name__ == "__main__":
    test_point_wise_feed_forward_network()
    
