import torch
import torch.nn as nn
from typing import Optional
import logging
from transformer.attention import MultiheadAttention
from transformer.utils import configure_device

LOGGER = logging.getLogger(__name__)
device = configure_device()


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.W_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.tensor) -> torch.tensor:
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
    def __init__(self, d_model: int, eps: float = 1e-5):
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

    def forward(self, x: torch.tensor) -> torch.float:
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

    def __init__(self, d_model: int, num_heads: int, d_ffn: int, dropout: float = 0.1):
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

    def forward(self, x: torch.tensor, mask: Optional[torch.tensor] = None):
        # Multihead self-attention sub-layer
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

    def __init__(self, d_model: int, num_heads: int, d_ffn: int, dropout: float = 0.1):
        super(DecoderLayer, self).__init__()

        # Self-attention sub-layer
        self.self_attention = MultiheadAttention(d_model, num_heads, dropout=dropout)

        # Encoder-Decoder attention sub-layer
        self.encoder_attention = MultiheadAttention(d_model, num_heads, dropout=dropout)

        # Position-wise feedforward sub-layer
        self.feedforward = PositionwiseFeedForward(d_model, d_ffn, dropout=dropout)

        # Layer normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.norm4 = LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.tensor,
        encoder_output: torch.tensor,
        self_mask: Optional[torch.tensor] = None,
        encoder_mask: Optional[torch.tensor] = None,
    ):
        # Self-attention sub-layer
        x_norm = self.norm1(x)
        self_attention_output, _ = self.self_attention(
            x_norm, x_norm, x_norm, mask=self_mask
        )
        x = x + self.dropout(self_attention_output)

        # Encoder-Decoder attention sub-layer
        x_norm = self.norm2(x)
        encoder_output_norm = self.norm3(encoder_output)
        encoder_attention_output, encoder_attention_weights = self.encoder_attention(
            x_norm, encoder_output_norm, encoder_output_norm, mask=encoder_mask
        )
        x = x + self.dropout(encoder_attention_output)

        # Position-wise feedforward sub-layer
        x_norm = self.norm4(x)
        ff_output = self.feedforward(x_norm)
        x = x + self.dropout(ff_output)

        return x, encoder_attention_weights


class Encoder(nn.Module):
    "Class for encoder, which consists of N-many EncoderLayers"

    def __init__(
        self,
        num_stacks: int,
        d_model: int,
        num_heads: int,
        d_ffn: int,
        dropout: float = 0.1,
    ):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model, num_heads=num_heads, d_ffn=d_ffn, dropout=dropout
                )
                for _ in range(num_stacks)
            ]
        )

    def forward(self, x: torch.tensor, mask: torch.tensor):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    "Class for decoder, which consists of N-many DecoderLayers"

    def __init__(
        self,
        num_stacks: int,
        d_model: int,
        num_heads: int,
        d_ffn: int,
        dropout: float = 0.1,
    ):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model, num_heads=num_heads, d_ffn=d_ffn, dropout=dropout
                )
                for _ in range(num_stacks)
            ]
        )

    def forward(
        self,
        x: torch.tensor,
        encoder_output: torch.tensor,
        self_mask: torch.tensor,
        encoder_mask: torch.tensor,
    ):
        "Pass the input (and mask) through each layer in turn."
        attn_weights = []
        for layer in self.layers:
            x, encoder_decoder_attn_weights = layer(
                x, encoder_output, self_mask, encoder_mask
            )
            attn_weights.append(encoder_decoder_attn_weights)
        return x, attn_weights


def positional_encoding(max_len: int, d_model: int):
    """
    Computes positional encoding according to
    PE(pos, 2i) = sin(pos/10000^{2i / dmodel})
    PE(pos, 2i + 1) = cos(pos/10000^{2i / dmodel})
    """
    div_terms = torch.pow(torch.tensor(10_000.0), torch.arange(0, d_model, 2) / d_model)
    pos_enc = torch.arange(max_len, dtype=torch.float32).unsqueeze(1).repeat(1, d_model)

    pos_enc[:, 0::2] = torch.sin(pos_enc[:, 0::2] / div_terms)
    pos_enc[:, 1::2] = torch.cos(pos_enc[:, 1::2] / div_terms)

    return pos_enc


class Embeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super(Embeddings, self).__init__()
        self.vocab_size = vocab_size
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.tensor):
        output = self.lut(x) * torch.sqrt(torch.tensor(self.d_model)).to(device)
        return output


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        num_encoder_stacks: int = 6,
        num_decoder_stacks: int = 6,
        d_model: int = 512,
        d_ffn: int = 2048,
        num_encoder_heads: int = 8,
        num_decoder_heads: int = 8,
        max_seq_len: int = 100,
        dropout: float = 0.1,
    ):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.src_embedding = Embeddings(src_vocab_size, d_model)
        self.tgt_embedding = Embeddings(tgt_vocab_size, d_model)
        self.encoder = Encoder(num_encoder_stacks, d_model, num_encoder_heads, d_ffn)
        self.decoder = Decoder(num_decoder_stacks, d_model, num_decoder_heads, d_ffn)
        # Mark positional encoder as not learnable, so that .parameters() doesn't pass it to optimizer
        self.register_buffer(
            "positional_encoder", positional_encoding(max_seq_len, d_model)
        )
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        self.src_dropout = nn.Dropout(dropout)
        self.tgt_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.tensor, src_mask: torch.tensor):
        LOGGER.debug(
            "Computing forward pass of encoder with "
            f"{src.size()=}, {src_mask.size()=}"
        )
        # Embed inputs, add position encoding, apply dropout
        src = self.src_embedding(src)
        src = src + self.positional_encoder[: src.size(1)]
        src = self.src_dropout(src)

        # Encode the source sequence
        enc_output = self.encoder(src, src_mask)
        return enc_output

    def decode(
        self,
        tgt: torch.tensor,
        enc_output: torch.tensor,
        tgt_mask: torch.tensor,
        src_mask: torch.tensor,
    ):
        LOGGER.debug(
            "Computing forward pass of decoder with "
            f"{tgt.size()=}, {enc_output.size()=}, {tgt_mask.size()=}, {src_mask.size()=}"
        )
        # Embed targets, add position encoding, apply dropout
        tgt = self.tgt_embedding(tgt)
        tgt = tgt + self.positional_encoder[: tgt.size(1)]
        tgt = self.tgt_dropout(tgt)

        # Decode the target sequence using the encoder output
        dec_output, encoder_decoder_attn_weights = self.decoder(
            tgt, enc_output, tgt_mask, src_mask
        )
        return dec_output, encoder_decoder_attn_weights

    def forward(
        self,
        src: torch.tensor,
        tgt: torch.tensor,
        tgt_mask: torch.tensor,
        src_mask: torch.tensor,
    ):
        """
        Forward pass of Transformer.

        - src has size (batch_size, src_seq_len)
        - tgt has size (batch_size, tgt_seq_len)
        - src_mask has size (batch_size, 1, seq_len), and
          prevents attention to padding indices
        - tgt_mask has size (1, tgt_seq_len, tgt_seq_len), and
          prevents attention to future positions and padding
        - output has size (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        LOGGER.debug(
            f"computing forward pass with {src.size()=} "
            f"{tgt.size()=} {src_mask.size()=} {tgt_mask.size()=}"
        )

        enc_output = self.encode(src, src_mask)
        dec_output, encoder_decoder_attn_weights = self.decode(
            tgt, enc_output, tgt_mask, src_mask
        )

        # Compute output layer
        logits = self.output_layer(dec_output)
        return logits, encoder_decoder_attn_weights
