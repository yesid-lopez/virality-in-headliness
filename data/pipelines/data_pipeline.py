import tensorflow as tf
from data.generators.pair import headlines_pair_generator


def create_data_pipeline(packages_ids, batch_sizes, max_length, encoder_fn, one_hot=False, classification=True,
                         cased=False, enforced=False):
    """
    Constructs a tf.data pipeline that yields batches of pairs of headlines with corresponding label (classification,
    regression).

    Args:
        packages_ids(dict): dictionary that maps dataset split to list of packages ids. Keys: ["train". "val", "test"] 
        batch_sizes(dict): dictionary that maps dataset split to batch size. Keys: ["train", "val", "test"]
        max_length(int): Number of maximum tokens. Required for padding sequences.
        encoder_fn(function): Function that maps a headline to its corresponding token list.
                              Example "This is just a headline example" -> [89, 3, 45, 65, 40, 78]
        one_hot(bool): whether to one hot encode class labels. Only valid if
               classification = True, otherwise a ValueError will be raised

        classification(bool): bool, whether to yield labels as a binary classification task.
                      1 ([0, 1] if one_hot=True) if first headline had more clicks than
                      the second headline, 0 ([1, 0] if one_hot=True) otherwise.

        cased(bool): If True, yielded headlines will not be lowercased.
             Otherwise, all headlines will be lowercased,

        enforced: If True, then the same pair of headlines are yielded twice,
                but the second pair is swapped  with labels and features (if valid)
                changed accordingly.
    Returns:
        train, val and test, all are BatchDataset instances.
    """

    kwargs = {
        "one_hot": one_hot,
        "classification": classification,
        "regression": False,
        "sorted": False,
        "features": None,
        "cased": cased,
        "enforced": enforced
    }

    train_package_ids = packages_ids["train"]
    val_package_ids = packages_ids["val"]
    test_package_ids = packages_ids["test"]

    def pad(x):
        return x + [0] * (max_length - len(x))

    def filter_label_and_tokenize(h1, h2, y_reg, y_class):
        tokens1, tokens2 = list(map(encoder_fn, [h1, h2]))
        padded_tokens1, padded_tokens2 = list(map(pad, [tokens1, tokens2]))

        inputs = {
            "input_tokens_h1": padded_tokens1,
            "input_tokens_h2": padded_tokens1
        }
        return inputs, y_reg, y_class

    def single_data_generator_wrapper(ids, **kwargs):
        generator = headlines_pair_generator(ids, **kwargs)
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
                            "input_tokens_h1": tf.TensorSpec(shape=(max_length,), dtype=tf.int32),
                            "input_tokens_h2": tf.TensorSpec(shape=(max_length,), dtype=tf.int32)
                        },
                        tf.TensorSpec(shape=(), dtype=tf.int32)
    )

    kwarg = {
        'output_signature': output_signature
    }

    train_data = tf.data.Dataset.from_generator(train_generator, **kwarg) \
        .prefetch(tf.data.AUTOTUNE) \
        .batch(batch_sizes["train"])

    val_data = tf.data.Dataset.from_generator(val_generator, **kwarg) \
        .prefetch(tf.data.AUTOTUNE) \
        .batch(batch_sizes["val"])

    test_data = tf.data.Dataset.from_generator(test_generator, **kwarg) \
        .prefetch(tf.data.AUTOTUNE) \
        .batch(batch_sizes["test"])

    return train_data, val_data, test_data
