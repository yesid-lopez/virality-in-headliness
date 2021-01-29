from headlines_project.lib import *


def plot_attention_weights(transformer_encoder, encoded_sentences_tokens, token_decoder, layer="encoder_layer_1"):
    """
    Plots self-attention weights for each head in the Transformer Encoder at a given layer

    Args:
       transformer_encoder(TransformerEncoder): TransformerEncoder instance
       encoded_sentences_tokens(numpy.array): (n, maxlen) of tokens where n is the number of headlines_project
       token_decoder(function): function that inputs a token and returns its corresponding word
       layer(str): name of the layer to extract attention weights from.

    Returns:
      A matplotlib Figure object with subplots of dimensions (N sentences, num_heads)
      where each plot is a heatmap of attention weights
    """
    num_heads = transformer_encoder.num_heads
    # number of headlines_project
    n = int(tf.shape(encoded_sentences_tokens)[0])

    _, att_weights = transformer_encoder(encoded_sentences_tokens)
    att_weights = att_weights[layer]  # (n or batch_size, num_heads, maxlen, maxlen)

    fig, axs = plt.subplots(nrows=n, ncols=num_heads, figsize=(4 * num_heads, 4 * n))
    fig.suptitle("Attention weights learned for each headline (winner headline in red)")

    for ith_headline in range(n):
        for jth_head in range(num_heads):
            # Don't consider padding tokens for visualize attention weights
            unpadded_tokens = list(filter(lambda x: x != 0, encoded_sentences_tokens[ith_headline, :]))
            decoded_subwords = [token_decoder(token) for token in unpadded_tokens]
            non_padding_tokens_len = len(decoded_subwords)

            # plot the attention weights
            ax = axs[ith_headline, jth_head] if num_heads > 1 else axs[ith_headline]
            att = att_weights[ith_headline, jth_head, :non_padding_tokens_len, :non_padding_tokens_len]
            ax.matshow(att, cmap='viridis')

            fontdict = {'fontsize': 8}

            ax.set_xticks(range(non_padding_tokens_len))
            ax.set_yticks(range(non_padding_tokens_len))
            # ax.set_ylim(non_padding_tokens_len, -0.5)
            ax.set_xticklabels(decoded_subwords, fontdict=fontdict, rotation=90)
            ax.set_yticklabels(decoded_subwords, fontdict=fontdict)
            ax.set_xlabel('Head {}'.format(jth_head + 1))

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig
