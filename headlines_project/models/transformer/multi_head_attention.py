from headlines_project.lib import *
from tensorflow.python.keras.layers import Dense


def scaled_dot_product_attention(queries, keys, values, mask=None):
    """
    Computes scaled dot product attention given queries, keys and values

    Args:
        queries(tf.Tensor): Tensor with shape == (batch_size, num_heads, seq_len, projection_dim)
        keys(tf.Tensor): Tensor with shape == (batch_size, num_heads, seq_len, projection_dim)
        values(tf.Tensor): Tensor with shape == (batch_size, num_heads, seq_len, projection_dim)
        mask(tf.Tensor): Tensor mask with ones in positions to avoid focusing on.

    Returns:
      output(tf.Tensor): shape(batch_size, num_heads, seq_len, projection_dim)
      attention_weights(tf.Tensor): shape (batch_size, num_heads, seq_len, seq_len)
    """
    dk = tf.cast(tf.shape(keys)[-1], tf.float32)
    # (batch_size, num_heads, seq_len, seq_len)
    att_logits = tf.matmul(queries, keys, transpose_b=True)
    scaled_att_logits = att_logits / tf.sqrt(dk)

    if mask is not None:
        scaled_att_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_att_logits, axis=-1)
    # (batch_size, num_heads, seq_len, projection_dim)
    output = attention_weights @ values
    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        if d_model % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {d_model} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = d_model // num_heads

        self.query_dense = Dense(d_model)
        self.key_dense = Dense(d_model)
        self.value_dense = Dense(d_model)

        self.combine_heads = Dense(d_model)

    def separate_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, projection_dim).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, projection_dim)

        Args:
            x(tf.Tensor): Tensor of shape (batch_size, seq_len, d_model)
            batch_size(int): Batch size

        Returns:
            Tensor x with shape (batch_size, num_heads, seq_len, projection_dim)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, queries, keys, values, mask=None):
        """
        Computes forward pass.

        Args:
            queries(tf.Tensor):
            keys(tf.Tensor:
            values(tf.Tensor):
            mask(tf.Tensor): Optional;
        Returns:
            output(tf.Tensor): Contextualized embeddings with shape  (batch_size, seq_len_q, d_model)
            att_weights(tf.Tensor): Tensor with attention weights of shape (batch_size, num_heads, seq_len, seq_len)
        """
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(queries)[0]

        query = self.query_dense(queries)  # (batch_size, seq_len, d_model)
        key = self.key_dense(keys)  # (batch_size, seq_len, d_model)
        value = self.value_dense(values)  # (batch_size, seq_len, d_model)

        query = self.separate_heads(query, batch_size)  # (batch_size, num_heads, seq_len_q, projection_dim)
        key = self.separate_heads(key, batch_size)  # (batch_size, num_heads, seq_len_k, projection_dim)
        value = self.separate_heads(value, batch_size)  # (batch_size, num_heads, seq_len_v, projection_dim)

        # scaled_att.shape == (batch_size, num_heads, seq_len_q, projection_dim)
        # att_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_att, att_weights = scaled_dot_product_attention(query, key, value, mask)

        scaled_att = tf.transpose(scaled_att, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, projection_dim)

        concat_attention = tf.reshape(scaled_att, (batch_size, -1, self.d_model))  # (batch_size, seq_len, d_model)

        output = self.combine_heads(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, att_weights


if __name__ == "__main__":
    from mask_utils import create_padding_mask, create_look_ahead_mask
    import matplotlib.pyplot as plt

    mha = MultiHeadAttention(d_model=100, num_heads=5)

    tokens = tf.constant([[1, 2, 0, 0, 0],
                          [3, 4, 5, 0, 0]])
    x = tf.random.uniform((tokens.shape[0], tokens.shape[1], 100))  # (batch_size, maxlen, d_model)

    padding_mask = create_padding_mask(tokens)
    look_ahead_mask = create_look_ahead_mask(tokens.shape[1])
    combined_mask = tf.maximum(padding_mask, look_ahead_mask)

    print(padding_mask.shape)
    out, att_weights = mha(x, x, x, mask=combined_mask)
    plt.matshow(att_weights[1, 0, :, :])  # Visualize causal self-attention in the first head for the second sample
    plt.show()
    print(out.shape, att_weights.shape)
