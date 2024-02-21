import torch
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from transformer.attention import compute_src_mask, future_mask
from torchtext.data.metrics import bleu_score
from typing import Optional
from transformer.utils import configure_device

LOGGER = logging.getLogger(__name__)
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = configure_device()


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
    model,
    src,
    sentence_ind,
    layer,
    attention_head,
    src_vocab,
    tgt_vocab,
    filename: Optional[str] = None,
):
    start_idx = src_vocab["<sos>"]
    end_idx = src_vocab["<eos>"]
    pad_idx = src_vocab["<blank>"]
    tgt_input, attention_head_weights = greedy_translate(
        model,
        src[0].unsqueeze(0),
        start_token=start_idx,
        end_token=end_idx,
        padding_token=pad_idx,
    )
    src_sentence = [src_vocab.lookup_token(elem.item()) for elem in src[0]]
    tgt_sentence = [tgt_vocab.lookup_token(elem) for elem in tgt_input]

    plt.figure(figsize=(12, 10))
    try:
        src_eos = src_sentence.index("<blank>")
    except ValueError:
        src_eos = len(src_sentence)

    tgt_eos = len(tgt_sentence)

    sns.heatmap(
        attention_head_weights[layer][0][attention_head][:tgt_eos, :src_eos]
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
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def tokens_to_string(
    tokens: torch.tensor, vocab, ignore_special_toks=False
) -> list[str]:
    """Convert tokens to sentence."""
    sos_idx = vocab["<sos>"]
    eos_idx = vocab["<eos>"]
    pad_idx = vocab["<blank>"]
    sentence = []
    for idx in tokens:
        # Handle both torch.tensors and list[int]
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        if ignore_special_toks and idx in [sos_idx, pad_idx, eos_idx]:
            continue

        sentence.append(vocab.lookup_token(idx))

    return sentence


def eval_model(model, tgt_vocab, test_dataloader, logger=None):
    """
    Collect model predictions and ground truth translations, then
    pass to a BLEU calculator.

    E.g.:
    Translation:  ['a', 'boy', 'raises', 'another', 'boy', 'on', 'to', 'his', 'back', '.']
    Ground truth: [['one', 'boy', 'hoists', 'another', 'boy', 'up', 'on', 'his', 'back', '.']]
    """
    model.eval()

    references: list[list[str]] = []
    hypotheses: list[list[list[str]]] = []
    start_token = tgt_vocab["<sos>"]
    end_token = tgt_vocab["<eos>"]
    pad_token = tgt_vocab["<blank>"]

    with torch.no_grad():
        for src, tgt, _ in test_dataloader:
            translations = []
            for elem in src:
                translation, _ = greedy_translate(
                    model, elem.unsqueeze(0), start_token, end_token, pad_token
                )
                translations.append(translation)

            # Convert output to actual text
            for i in range(len(translations)):
                translation = tokens_to_string(
                    translations[i], tgt_vocab, ignore_special_toks=True
                )  # Convert indices to text
                reference = tokens_to_string(
                    tgt[i], tgt_vocab, ignore_special_toks=True
                )  # Get the actual target sentence

                hypotheses.append(translation)
                references.append([reference])  # BLEU expects a list of references

    # Compute BLEU (using torchtext or sacrebleu)
    bleu = bleu_score(hypotheses, references)
    for translation, ground_truth in zip(hypotheses, references):
        LOGGER.info(f"Translation: {translation}")
        LOGGER.info(f"Ground truth: {ground_truth}")
    LOGGER.info(f"BLEU score: {bleu*100:.2f}")
    return bleu


def plot_alignment(alignment, src_sentence, tgt_sentence, filename):
    fig, ax = plt.subplots()
    cax = ax.matshow(alignment, cmap="gray")
    plt.yticks(range(len(tgt_sentence)), tgt_sentence, rotation=90)
    plt.xticks(range(len(src_sentence)), src_sentence)

    # Add colorbar for reference
    plt.colorbar(cax)

    plt.title("Alignment Visualization")
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def greedy_translate(model, src, start_token, end_token, padding_token, max_len=50):
    """
    Perform greedy translation.
    - src: Input source sequence tensor.
    - src_mask: Mask for the source sequence.
    - max_len: Maximum length of the generated translation.
    - start_token: Index of the start-of-sequence token.
    - end_token: Index of the end-of-sequence token.

    Returns:
    - translated_tokens: List of token indices for the generated translation.
    """
    model.eval()
    with torch.no_grad():
        # Encode the source sequence
        src_mask = compute_src_mask(src, padding_token)
        src_mask = src_mask.to(device)
        enc_output = model.encode(src, src_mask)

        # Initialize the target sequence with the start token
        tgt = torch.full((1, 1), start_token).to(device)

        for _ in range(max_len):
            # Generate the next token
            tgt_mask = future_mask(tgt.size(1))
            tgt_mask = tgt_mask.to(device)
            dec_output, attn_weights = model.decode(tgt, enc_output, tgt_mask, src_mask)
            dec_output = model.output_layer(dec_output)
            next_token = dec_output[:, -1, :].argmax(dim=-1).unsqueeze(1)

            # Append the generated token to the target sequence
            tgt = torch.cat([tgt, next_token], dim=1)

            # Stop if the end token is generated
            if next_token.item() == end_token:
                break
    # Convert tensor to list of token indices
    translated_tokens = tgt.squeeze().tolist()
    return translated_tokens, attn_weights
