# Note that losses and accuracies in this package is 
# batch loss and batch accuracy.


import tensorflow as tf

# the last dense layer of transfromer has no activation function,
# so you use from_logits = True
# losses should be masked to ignore the padding positions.
# you need to do the reduction manully, so you use reducetion = 'none'
# and leave the reduction step for yourself.
# this loss object returns a tensor instead of a scalar.
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction = 'none')


def loss_function(real, pred):

    """Desc: you should retreive the mask information from real tensor..
    use mask to filterout the padding positions.
    compute loss with no reduction.
    perform mask on result
    then reduction over non-padding positions.

    Parameters:
    ------------
    read: (N, T) shaped tensor
    pred: (N, T, V) shaped tensor


    Returns:
    ------------
    something like a float number
    """

    loss_ = loss_object(real, pred)

    # element wise when != 0, 1 else 0 
    # all non-padding positions is marked with 1
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=loss_.dtype)

    # then padding position loss if filtered out
    loss_ *= mask

    # reduction step:
    # tf.reduce_sum(loss_) = total loss in non-padding positions [in the batch] 
    # tf.reduce_sum(msak) = how many (non-padding) positions [in the batch]
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    
def accuracy_function(real, pred):

    # pred is of shpae(N, T, V), argmax over V to get the classIds
    # the result is of shape(N, T)
    prediction = tf.math.argmax(pred, axis = -1)

    # compare real and prediction elementsize (both have shape(N, T))
    ele_acc = tf.math.equal(real, prediction)

    # masking out padding positions
    mask = tf.logical_not(tf.math.equal(real, 0))

    # non-padding positions binary accuracy
    ele_acc = tf.math.logical_and(ele_acc, mask)
    ele_acc = tf.cast(ele_acc, tf.float32)
    mask = tf.cast(mask, tf.float32)

    return tf.reduce_sum(ele_acc) / tf.reduce_sum(mask)

def test_loss_function(): 
    pred = tf.random.uniform((10,15,3))
    real = tf.math.argmax(pred, axis = -1)
    print(loss_function(real, pred))
    print("test loss function success")

def test_accuracy_function():
    pred = tf.random.uniform((10,15,3))
    real = tf.math.argmax(pred, axis = -1)
    real = tf.cast(real, tf.int64)
    print(accuracy_function(real, pred))
    print("test loss function success")


if __name__ == "__main__":
    test_loss_function()
    test_accuracy_function()




