import torch
import torch.nn.functional as F
import torch.nn as nn
import logging

LOGGER = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_heads(Q, num_heads):
    return torch.stack(Q.split(num_heads, dim=-1))


class MultiheadAttention(nn.Module):
    """
    Class to compute multihead attention with num_heads-many heads
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear projection for the attention heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Linear projection for the output layer
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        LOGGER.debug(
            f"Computing multihead attention with {Q.size()=} {K.size()=} {V.size()=}"
        )
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        batch_size = Q.size(0)
        d_model = Q.size(-1)

        # Split into multiple heads
        Q = split_heads(Q, self.num_heads)
        K = split_heads(K, self.num_heads)
        V = split_heads(V, self.num_heads)

        # Compute attention
        output, attention_weights = attention(Q, K, V, dropout=self.dropout, mask=mask)
        # Concatenate the heads and compute transformation
        output = output.permute(0, 2, 1, 3).reshape(batch_size, -1, d_model)
        output = self.W_o(output)

        return output, attention_weights


def attention(Q, K, V, dropout=None, mask=None):
    """
    Computes attention given query, keys, values.
    If we have n-many key-value pairs of dimension dk, dv respectively
    and m-many queries of dimension dk, then

    - Q has shape batch_size \\times m \\times dk
    - K has shape batch_size \\times n \\times dk
    - V has shape batch_size \\times n \\times dv
    In the transformer architecture,
    - m = n = sequence_length
    - dk= dv = dmodel = 512.
    """
    LOGGER.debug(
        f"computing attention with dimensions {Q.size()=} {K.size()=} {V.size()=}"
        f" with mask.size()={mask.size() if mask is not None else None}"
    )
    dk = Q.size(-1)

    # Compute attention
    scale = torch.sqrt(torch.tensor(dk)).to(device)
    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

    # Apply attention mask (if provided).
    if mask is not None:
        LOGGER.debug(f"Applying {mask.size()=} to {attention_scores.size()=}")
        attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

    attention_weights = F.softmax(attention_scores, dim=-1)
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # Calculate the weighted sum of values
    LOGGER.debug(f"Attempting to multiply {attention_weights.size()=} with {V.size()=}")
    attended_values = torch.matmul(attention_weights, V)

    return attended_values, attention_weights


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.W_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Computes
        FFN(x_i) = ReLU(x_iW_1 + b_1)W_2 + b_2.

        - x has shape batch_size \\times seq_length \\times d_model
        """

        output = self.W_1(x)
        output = self.dropout(self.relu(output))
        output = self.W_2(output)

        return output


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        """
        Computes layer normalization.

        LayerNorm(x) =
        \\gamma \cdot \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta
        where
        - \\gamma is a scale parameter
        - \\mu is the mean
        - \\sigma is the standard deviation
        - \\epsilon is an offset for numerical stability
        - \\beta is a shift parameter.
        For training purposes \\sqrt{\\sigma^2 + \\epsilon} ~= \\sigma + \\epsilon.
        """
        super(LayerNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps

        # Learnable scale and shift parameters
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # Calculate mean and standard deviation along the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        # Apply LayerNorm formula
        x_normalized = self.gamma * (x - mean) / (std + self.eps) + self.beta

        return x_normalized


class EncoderLayer(nn.Module):
    """
    Implements a single Encoder layer with pre-layer normalization.
    """

    def __init__(self, d_model, num_heads, d_ffn, dropout=0.1):
        super(EncoderLayer, self).__init__()

        # Self-attention sub-layer
        self.self_attention = MultiheadAttention(d_model, num_heads, dropout=dropout)

        # Position-wise feedforward sub-layer
        self.feedforward = PositionwiseFeedForward(d_model, d_ffn, dropout=dropout)

        # Layer Normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multihead self-attention sub-layer
        LOGGER.debug(f"Computing forward pass of encoder layer with {x.size()=}")
        x_norm = self.norm1(x)
        attention_output, _ = self.self_attention(x_norm, x_norm, x_norm, mask=mask)
        x = x + self.dropout(attention_output)

        # Position-wise feedforward sub-layer
        x_norm = self.norm2(x)
        ff_output = self.feedforward(x_norm)
        output = x + self.dropout(ff_output)

        return output


class DecoderLayer(nn.Module):
    """
    Implements a single Decoder layer with pre-layer normalization.
    """

    def __init__(self, d_model, num_heads, d_ffn, dropout=0.1):
        super(DecoderLayer, self).__init__()

        # Self-attention sub-layer
        self.self_attention = MultiheadAttention(d_model, num_heads, dropout=dropout)

        # Encoder-Decoder attention sub-layer
        self.encoder_attention = MultiheadAttention(d_model, num_heads, dropout=dropout)

        # Position-wise feedforward sub-layer
        self.feedforward = PositionwiseFeedForward(d_model, d_ffn, dropout=dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, self_mask=None, encoder_mask=None):
        # Self-attention sub-layer
        x_norm = self.norm1(x)
        self_attention_output, _ = self.self_attention(
            x_norm, x_norm, x_norm, mask=self_mask
        )
        x = x + self.dropout(self_attention_output)

        # Encoder-Decoder attention sub-layer
        x_norm = self.norm2(x)
        encoder_output_norm = self.norm2(x)
        encoder_attention_output, _ = self.encoder_attention(
            encoder_output_norm, encoder_output_norm, x_norm, mask=encoder_mask
        )
        x = x + self.dropout(encoder_attention_output)

        # Position-wise feedforward sub-layer
        x_norm = self.norm3(x)
        ff_output = self.feedforward(x_norm)
        x = x + self.dropout(ff_output)

        return x


class Encoder(nn.Module):
    "Class for encoder, which consists of N-many EncoderLayers"

    def __init__(self, num_stacks, d_model, num_heads, d_ffn, dropout=0.1):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model, num_heads=num_heads, d_ffn=d_ffn, dropout=dropout
                )
                for _ in range(num_stacks)
            ]
        )

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    "Class for decoder, which consists of N-many DecoderLayers"

    def __init__(self, num_stacks, d_model, num_heads, d_ffn, dropout=0.1):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model, num_heads=num_heads, d_ffn=d_ffn, dropout=dropout
                )
                for _ in range(num_stacks)
            ]
        )

    def forward(self, x, encoder_output, self_mask, encoder_mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, encoder_output, self_mask, encoder_mask)
        return x


def positional_encoding(max_len, d_model):
    """
    Computes positional encoding according to
    PE(pos, 2i) = sin(pos/10000^{2i / dmodel})
    PE(pos, 2i + 1) = cos(pos/10000^{2i / dmodel})
    """
    div_terms = torch.pow(torch.tensor(10_000.0), torch.arange(0, d_model, 2) / d_model)
    pos_enc = (
        torch.arange(max_len, dtype=torch.float32).repeat(d_model, 1).transpose(-1, -2)
    )

    # Compute the sinusoidal positional encoding
    num_even_terms = len(div_terms)
    num_odd_terms = d_model - num_even_terms
    pos_enc[:, 0::2] = torch.sin(pos_enc[:, 0::2] / div_terms[:num_even_terms])
    pos_enc[:, 1::2] = torch.cos(pos_enc[:, 1::2] / div_terms[:num_odd_terms])

    return pos_enc


def subsequent_mask(d_model):
    """
    Creates a lower-triangular n \\times n matrix
    used to mask future positions
    """
    attn_shape = (1, d_model, d_model)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embeddings, self).__init__()
        self.vocab_size = vocab_size
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        LOGGER.debug(
            f"Computing {self.vocab_size=}, {self.d_model=} embedding with {x.size()=}"
        )
        output = self.lut(x) * torch.sqrt(torch.tensor(self.d_model))
        LOGGER.debug(f"Computed {output.size()=} embedding")
        return output


class Transformer(nn.Module):
    def __init__(
        self,
        num_encoder_stacks,
        num_decoder_stacks,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        d_ffn=2048,
        num_encoder_heads=8,
        num_decoder_heads=8,
        max_seq_len=100,
        dropout=0.1,
    ):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.src_embedding = Embeddings(src_vocab_size, d_model)
        self.tgt_embedding = Embeddings(tgt_vocab_size, d_model)
        self.encoder = Encoder(num_encoder_stacks, d_model, num_encoder_heads, d_ffn)
        self.decoder = Decoder(num_decoder_stacks, d_model, num_decoder_heads, d_ffn)
        self.positional_encoder = positional_encoding(max_seq_len, d_model)
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        self.src_dropout = nn.Dropout(dropout)
        self.tgt_dropout = nn.Dropout(dropout)

        # Initialize params with Glorot / fan_avg?

    def encode(self, src, src_mask):
        # Embed inputs, add position encoding, apply dropout
        src = self.src_embedding(src)
        src = src + self.positional_encoder[: src.size(1)]
        src = self.src_dropout(src)
        LOGGER.debug(f"Computing forward pass of encoder layer with {src.size()=}")

        # Encode the source sequence
        enc_output = self.encoder(src, src_mask)
        return enc_output

    def decode(self, tgt, enc_output, tgt_mask, src_mask):
        # Embed targets, add position encoding, apply dropout
        tgt = self.tgt_embedding(tgt)
        tgt = tgt + self.positional_encoder[: tgt.size(1)]
        tgt = self.tgt_dropout(tgt)

        # Decode the target sequence using the encoder output
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        return dec_output

    def forward(self, src, tgt, src_mask, tgt_mask):
        LOGGER.debug(
            f"computing forward pass with {src.size()=} "
            f"{tgt.size()=} {src_mask.size()=} {tgt_mask.size()=}"
        )

        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, tgt_mask, src_mask)

        # Compute output layer
        output = self.output_layer(dec_output)
        output = torch.softmax(output, dim=-1)
        LOGGER.debug(f"Returning {output=} of {output.size()=}")

        return output

