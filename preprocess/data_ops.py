import tensorflow as tf
import tensorflow_datasets as tfds
from functools import partial


def get_data(data_dir = "./tensorflow_data"):
    examples, _ = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                                as_supervised=True, data_dir = data_dir)
    train_examples, val_examples = examples['train'], examples['validation']
    return train_examples, val_examples

def get_tokenizers(cache_dir = "."):
    model_name = 'ted_hrlr_translate_pt_en_converter'
    tf.keras.utils.get_file(
        f'{model_name}.zip',
        f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
        cache_dir=cache_dir, cache_subdir='', extract=True)

    tokenizers = tf.saved_model.load(model_name)
    return tokenizers

def filter_token_by_length(max_tokens):
    def filter_max_tokens(pt, en):
        num_tokens = tf.maximum(tf.shape(pt)[1],tf.shape(en)[1])
        return num_tokens < max_tokens
    return filter_max_tokens

def tokenizer_pair_with_tokenizers(tokenizers):
    def tokenize_pairs(pt, en):
        pt = tokenizers.pt.tokenize(pt)
        # Convert from ragged to dense, padding with zeros.
        pt = pt.to_tensor()

        en = tokenizers.en.tokenize(en)
        # Convert from ragged to dense, padding with zeros.
        en = en.to_tensor()
        return pt, en
    return tokenize_pairs

def dataset_creator(ds, batch_size, buffer_size, tokenizers, max_tokens):
  return (
      ds
      .cache()
      .shuffle(buffer_size)
      .batch(batch_size)
      .map(tokenizer_pair_with_tokenizers(tokenizers), num_parallel_calls=tf.data.AUTOTUNE)
      .filter(filter_token_by_length(max_tokens))
      .prefetch(tf.data.AUTOTUNE))


make_batches = partial(
    dataset_creator, 
    batch_size = 64, 
    buffer_size = 20000, 
    tokenizers = get_tokenizers(), 
    max_tokens = 128)