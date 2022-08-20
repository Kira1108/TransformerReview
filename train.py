import tensorflow as tf
from model.transformer import Transformer
from preprocess import get_tokenizers, dataset_creator, get_data
from schedule import PaperScheduler
from losses import loss_function, accuracy_function
import time
from functools import partial

tokenizers = get_tokenizers()

# hypter parameters, you can optimize
NUM_LAYERS = 4
D_MODEL = 128
DFF = 512
NUM_HEADS = 8
DROPOUT_RATE = 0.1
MAX_TOKENS = 128

# training parameters
EPOCHS = 20
BUFFER_SIZE = 20000
BATCH_SIZE = 64



# hypterparameter from tokenizers
INPUT_VOCAB_SIZE = tokenizers.pt.get_vocab_size().numpy()
TARGET_VOCAB_SIZE = tokenizers.en.get_vocab_size().numpy()

# transformer model
transformer = Transformer(
    num_layers = NUM_LAYERS,
    input_vocab_size = INPUT_VOCAB_SIZE, 
    target_vocab_size = TARGET_VOCAB_SIZE,
    d_model = D_MODEL,
    max_tokens = MAX_TOKENS,
    num_heads = NUM_HEADS,
    dff = DFF,
    rate = DROPOUT_RATE
)

learning_rate = PaperScheduler(D_MODEL)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

checkpoint_path = './checkpoints/train'

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]



# tf.function take functions into a graph, that can execute faster on distributed learning case.
@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    """train_step is one step on a single batch, with forward and backward propagation."""
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    with tf.GradientTape() as tape:
        predictions, _ = transformer([inp, tar_inp],
                                        training = True)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, predictions))

make_batches = partial(
    dataset_creator,
    batch_size = BATCH_SIZE,
    buffer_size = BUFFER_SIZE,
    tokenizers = tokenizers,
    max_tokens = MAX_TOKENS)

train_examples, val_examples = get_data()
train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    # inp -> portuguese, tar -> english
    for (batch, (inp, tar)) in enumerate(train_batches):
        train_step(inp, tar)

        if batch % 50 == 0:
            print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
        print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')