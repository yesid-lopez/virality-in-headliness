from headlines_project.lib import *
from .mask_utils import create_look_ahead_mask, create_padding_mask


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, token_embedding):
        """
        Layer for converting tokens ids into embeddings with positional information encoded
        and create padding and look ahead masks (optional)

        :param token_embedding: (tf.keras.layers.Embedding): Embedding layer
        """
        super(PositionalEmbedding, self).__init__()
        self.vocab_size = token_embedding.input_dim
        self.d_model = token_embedding.output_dim
        self.max_length = token_embedding.input_length
        self.token_emb = token_embedding
        self.pos_encoding = positional_encoding(self.vocab_size, self.d_model)

    def call(self, input_tokens, training, causal_attention=False):
        """ 
        Computes embeddings for input tokens ids and apply them positional encoding
        Args:
          input_tokens(tensor): tensor with shape (batch_size, max_length)
          causal_attention(bool): If True, padding mask is combined with look ahead mask. 
                                  Otherwise only padding mask is created. 

        Returns:
          x: Positional encoded embeddings 
          mask: Padding mask for Multi Head Attention (with lookahead, optionally)
        """
        max_length = tf.shape(input_tokens)[-1]
        mask = create_padding_mask(input_tokens)
        if causal_attention:
            look_ahead_mask = create_look_ahead_mask(max_length)
            mask = tf.maximum(mask, look_ahead_mask)

        x = self.token_emb(input_tokens, training=training)  # (batch_size, max_length, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :max_length, :]
        return x, mask


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    pos = 10000
    d_model = 100
    pos_encoding = positional_encoding(pos, d_model)
    print(pos_encoding.shape)

    plt.pcolormesh(pos_encoding[0][:50, :], cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, d_model))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()
