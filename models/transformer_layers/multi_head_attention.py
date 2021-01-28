import tensorflow as tf
from tensorflow.keras.layers import Dense


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

    @staticmethod
    def scaled_dot_product_attention(queries, keys, values, mask=None):
        """
        Arguments:
        query: Query shape == (batch_size, num_heads, seq_len, projection_dim)
        key: Key shape == (batch_size, num_heads, seq_len, projection_dim)
        value: Value shape == (batch_size, num_heads, seq_len, projection_dim)

        Returns:
          output:            (batch_size, num_heads, seq_len, projection_dim) 
          attention_weights: (batch_size, num_heads, seq_len, seq_len)
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

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, queries, keys, values, mask=None):
        """ 
          output:  (batch_size, seq_len, d_model)
          weights: (batch_size, num_heads, seq_len, seq_len)
        """
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(queries)[0]
        query = self.query_dense(queries)  # (batch_size, seq_len, d_model)
        key = self.key_dense(keys)  # (batch_size, seq_len, d_model)
        value = self.value_dense(values)  # (batch_size, seq_len, d_model)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.scaled_dot_product_attention(
            query, key, value, mask)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.d_model)
        )  # (batch_size, seq_len, d_model)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, num_heads, seq_len, seq_len)
        return output, weights
