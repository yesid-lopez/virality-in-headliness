from headlines_project.lib import *


def create_padding_mask(seq, padding_token=0):
    """
    Creates a mask with ones where padding_token is found and zeros elsewhere.
    This mask is added to the attention logits in order to prevent the 
    model to focus in padding tokens.

    Args:
        seq: tensor with shape (batch_size, max_length)
        padding_token: token for padding, Default: 0.
    Returns:
        mask of shape (batch_size, 1, 1, max_length)
    """
    seq = tf.cast(tf.math.equal(seq, padding_token), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits and match dimensions.
    mask = seq[:, tf.newaxis, tf.newaxis, :]
    return mask  # (batch_size, 1, 1, max_length)


def create_look_ahead_mask(max_length):
    """
    Creates mask to causal attention. 
    This mask will prevent the model to focus on future tokens.

    Args:
        max_length: max_length of the sequence

    Returns:
        Tensor of shape (max_length, max_length) with its upper filled with
        ones and zeros elsewhere
    """
    mask = 1 - tf.linalg.band_part(tf.ones((max_length, max_length)), -1, 0)
    return mask


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    tokens = tf.constant([[743, 623, 0, 0, 0],
                          [132, 243, 453, 0, 0],
                          [365, 400, 290, 265, 509]])
    att_logits = tf.random.uniform((3, 1, 5, 5))  # Simulate att logits (batch_size, num_heads, maxlen, maxlen)
    padding_mask = create_padding_mask(tokens)
    print(f"padding_mask: {padding_mask}")
    masked_att_logits = att_logits + (padding_mask * -1e9)
    att_weights = tf.nn.softmax(masked_att_logits, axis=-1)
    plt.matshow(att_weights[1, 0, :, :].numpy())
    plt.show()
