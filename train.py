import tensorflow as tf
from model.transformer import Transformer
from preprocess import get_tokenizers
from schedule import PaperScheduler
from losses import loss_function, accuracy_function

tokenizers = get_tokenizers()

# hypter parameters, you can optimize
NUM_LAYERS = 4
D_MODEL = 128
DFF = 512
NUM_HEADS = 8
DROPOUT_RATE = 0.1
MAX_TOKENS = 128

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