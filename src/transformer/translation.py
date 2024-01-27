import torch
import seaborn as sns
import matplotlib.pyplot as plt


def top_attentions(attention_matrix, num_tops=3):
    """
    Find positions of the top three highest attention values for each head.

    :param attention_matrix: A tensor of shape (num_heads, src_seq_len, tgt_seq_len)
    :return: A list of tuples containing the positions of the top three values for each head.
            Each tuple contains (head_index, src_index, tgt_index).
    """
    num_heads, src_seq_len, tgt_seq_len = attention_matrix.shape
    top_positions = []

    # Iterate over each head
    for head_index in range(num_heads):
        # Flatten the attention matrix for the current head to find top values
        flat_attention = attention_matrix[head_index].view(-1)

        # Get the indices of the top 3 values
        top_values, top_indices = torch.topk(flat_attention, num_tops)

        # Convert flat indices to 2D indices (src_index, tgt_index)
        top_positions_head = [
            (
                head_index,
                int(index / tgt_seq_len),
                int(index % tgt_seq_len),
                value.item(),
            )
            for index, value in zip(top_indices, top_values)
        ]
        top_positions.extend(top_positions_head)

    return top_positions


def determine_alignments(src_vocab, tgt_vocab, attention_matrix):
    top_alignments = {}
    for alignment in top_attentions(attention_matrix):
        if top_alignments.get(alignment[0]) is None:
            top_alignments[alignment[0]] = []
        top_alignments[alignment[0]].append(
            {
                "source_word": src_vocab.lookup_token(alignment[1]),
                "target_word": tgt_vocab.lookup_token(alignment[2]),
                "score": alignment[3],
            }
        )

    return top_alignments


def plot_attention(
    attention_head_weights: torch.tensor,
    sentence_ind,
    layer,
    attention_head,
    src_sentence: list[str],
    tgt_sentence: list[str],
):
    plt.figure(figsize=(10, 8))
    src_eos = src_sentence.index("<blank>")
    tgt_eos = src_sentence.index("<blank>")
    sns.heatmap(
        attention_head_weights[layer][sentence_ind][attention_head][:src_eos, :tgt_eos]
        .detach()
        .cpu()
        .numpy(),
        annot=True,
        cmap="viridis",
        xticklabels=src_sentence[:src_eos],
        yticklabels=tgt_sentence[:tgt_eos],
    )
    plt.xlabel("Source Sentence")
    plt.ylabel("Target Sentence")
    plt.title(
        f"Encoder-Decoder {layer} Attention Head {attention_head}, Sequence {sentence_ind}"
    )
    plt.show()
