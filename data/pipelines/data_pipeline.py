import tensorflow as tf
from data.generators.data_generator import single_data_generator

def get_datasets_for_transformer(packages, batch_sizes, maxlen, encoder_fn, one_hot=False, classification=True,
                                 regression=False, sorted=False, cased=False, enforced=False):
    """
    Constructs a tf.data pipeline that yields batches of pairs of headlines with corresponding label (classification,
    regression).

    :param packages: (dict) Dictionary
    :param batch_sizes:
    :param maxlen:
    :param encoder_fn:
    :param one_hot:
    :param classification:
    :param regression:
    :param sorted:
    :param cased:
    :param enforced:
    :return:
    """

    train_package_ids = packages['train_package_ids']
    val_package_ids = packages['val_package_ids']
    test_package_ids = packages["test_package_ids"]

    kwargs = {
        "one_hot": one_hot,
        "classification": classification,
        "regression": regression,
        "sorted": sorted,
        "features": None,
        "cased": cased,
        "enforced": enforced
    }

    def filter_label_and_tokenize(h1, h2, y_reg, y_class):
        tokens1, tokens2 = list(map(encoder_fn, [h1, h2]))
        pad = lambda x: x + [0] * (maxlen - len(x))

        h1 = pad(tokens1)
        h2 = pad(tokens2)

        inputs = {
            "input_tokens_h1": h1,
            "input_tokens_h2": h2
        }
        return inputs, y_class

    def single_data_generator_wrapper(ids, **kwargs):
        generator = single_data_generator(ids, **kwargs)
        for inputs in generator:
            inputs, y_class = filter_label_and_tokenize(*inputs)
            yield inputs, y_class

    def train_generator():
        return single_data_generator_wrapper(train_package_ids, **kwargs)

    def val_generator():
        kwargs["enforced"] = False
        return single_data_generator_wrapper(val_package_ids, **kwargs)

    def test_generator():
        kwargs["enforced"] = False
        return single_data_generator_wrapper(test_package_ids, **kwargs)

    output_signature = ({
                            "input_tokens_h1": tf.TensorSpec(shape=(maxlen,), dtype=tf.int32),
                            "input_tokens_h2": tf.TensorSpec(shape=(maxlen,), dtype=tf.int32)
                        },
                        tf.TensorSpec(shape=(), dtype=tf.int32)
    )

    kwarg = {
        'output_signature': output_signature
    }

    train_data = tf.data.Dataset.from_generator(train_generator, **kwarg) \
                                .prefetch(tf.data.AUTOTUNE)\
                                .batch(batch_sizes["train"])

    val_data = tf.data.Dataset.from_generator(val_generator, **kwarg) \
                              .prefetch(tf.data.AUTOTUNE) \
                              .batch(batch_sizes["val"])

    test_data = tf.data.Dataset.from_generator(test_generator, **kwarg) \
                               .prefetch(tf.data.AUTOTUNE)\
                               .batch(batch_sizes["test"])

    return train_data, val_data, test_data
