import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout
from transformer_layers.multi_head_attention import MultiHeadAttention
from transformer_layers.positional_embedding import PositionalEmbedding


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, rate=0.2):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(d_model),
            ]
        )
        self.layer_norm1 = LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training, mask=None):
        """
        output # (batch_size, seq_len, d_model) 
        weights # (batch_size, num_heads, seq_len, seq_len)
        """
        attn_output, weights = self.mha(inputs, inputs, inputs, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        output = self.layer_norm2(out1 + ffn_output)
        return output, weights


class TransformerEncoder(tf.keras.models.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, embedding_layer,
                 causal_attention=False, dropout_rate=0.2):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding_layer = embedding_layer
        self.encoder_layers = [TransformerEncoderLayer(d_model, num_heads, dff, rate=dropout_rate)
                               for _ in range(num_layers)]
        error_message = f"d_model and embedding output dim must be equal, {d_model} != {self.embedding_layer.output_dim}"
        assert d_model == self.embedding_layer.output_dim, error_message
        self.positional_embedding = PositionalEmbedding(self.embedding_layer)
        self.causal_attention = causal_attention
        self.dropout = Dropout(dropout_rate)

    def call(self, input_tokens, training):
        """
        Args:
          input_tokens: tensor with shape (batch_size, max_length) of tokens
        Returns:
          x: (batch_size, input_seq_len, d_model)
          layers_att_weights: dictionary with layer names as keys and 
                              attention weights of shape (batch_size, num_heads, max_length, max_length)
                              for each layer as values 
        """
        embeddings, mask = self.positional_embedding(input_tokens, training=training,
                                                     causal_attention=self.causal_attention)

        x = self.dropout(embeddings, training=training)

        layers_att_weights = {}

        for i in range(self.num_layers):
            x, weights = self.encoder_layers[i](x, training, mask)
            layers_att_weights[f"encoder_layer_{i + 1}"] = weights
        return x, layers_att_weights