from model.transformer import Transformer

# hypter parameters
NUM_LAYERS = 4
D_MODEL = 128
DFF = 512
NUM_HEADS = 8
DROPOUT_RATE = 0.1
INPUT_VOCAB_SIZE = 1000
TARGET_VOCAB_SIZE = 1000
MAX_TOKENS = 256
DFF = 2048

# transformer model
model = Transformer(
    num_layers = NUM_LAYERS,
    input_vocab_size = INPUT_VOCAB_SIZE, 
    target_vocab_size = TARGET_VOCAB_SIZE,
    d_model = D_MODEL,
    max_tokens = MAX_TOKENS,
    num_heads = NUM_HEADS,
    dff = DFF,
    rate = DROPOUT_RATE
)
