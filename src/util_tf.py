from copy import copy
from util import Record
import tensorflow as tf


scope = tf.variable_scope


def profile(sess, wtr, run, feed_dict= None, prerun= 3, tag= 'flow'):
    for _ in range(prerun): sess.run(run, feed_dict)
    meta = tf.RunMetadata()
    sess.run(run, feed_dict, tf.RunOptions(trace_level= tf.RunOptions.FULL_TRACE), meta)
    wtr.add_run_metadata(meta, tag)


def pipe(*args, prefetch= 1, repeat= -1, name= 'pipe', **kwargs):
    """see `tf.data.Dataset.from_generator`."""
    with tf.variable_scope(name):
        return tf.data.Dataset.from_generator(*args, **kwargs) \
                              .repeat(repeat) \
                              .prefetch(prefetch) \
                              .make_one_shot_iterator() \
                              .get_next()


def placeholder(dtype, shape, x= None, name= None):
    """returns a placeholder with `dtype` and `shape`.

    if tensor `x` is given, converts and uses it as default.

    """
    if x is None: return tf.placeholder(dtype, shape, name)
    try:
        x = tf.convert_to_tensor(x, dtype)
    except ValueError:
        x = tf.cast(x, dtype)
    return tf.placeholder_with_default(x, shape, name)


def trim(x, eos, name= 'trim'):
    """trims a tensor of sequences

    x   : tensor i32 (?, b)
    eos : tensor i32 ()
       -> tensor i32 (t, b)  the trimmed sequence tensor
        , tensor b8  (t, b)  the sequence mask
        , tensor i32 (b,)    the sequence lengths

    each column aka sequence in `x` is assumed to be any number of
    non-eos followed by any number of eos

    """
    with scope(name):
        with scope('not_eos'): not_eos = tf.not_equal(x, eos)
        with scope('len_seq'): len_seq = tf.reduce_sum(tf.to_int32(not_eos), axis=0)
        with scope('max_len'): max_len = tf.reduce_max(len_seq)
        return x[:max_len], not_eos[:max_len], len_seq


def get_shape(x, name= 'shape'):
    """returns the shape of `x` as a tuple of integers (static) or int32
    scalar tensors (dynamic)

    """
    with scope(name):
        shape = tf.shape(x)
        shape = tuple(d if d is not None else shape[i] for i, d in enumerate(x.shape.as_list()))
        return shape
