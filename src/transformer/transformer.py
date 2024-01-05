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

    def __init__(
        self,
        d_model,
        num_heads,
    ):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear projection for the attention heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Linear projection for the output layer
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, Q, K, V):
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        # Split into multiple heads
        Q = split_heads(Q, self.num_heads)
        K = split_heads(K, self.num_heads)
        V = split_heads(V, self.num_heads)

        # Compute attention
        output, attention_weights = attention(Q, K, V)
        # Concatenate the heads and compute transformation
        output = output.permute(0, 2, 1, 3).reshape(2, 4, -1)
        output = self.W_o(output)

        return output, attention_weights


def multihead_attention(Q, K, V, W_q, W_k, W_v):
    """
    Computes self-attention

    - W_q has shape dmodel \\times d_k
    - W_k has shape dmodel \\times d_k
    - W_v has shape dmodel \\times d_v
    """
    # Linear projections
    Q_proj = torch.matmul(Q, W_q)
    K_proj = torch.matmul(K, W_k)
    V_proj = torch.matmul(V, W_v)

    # Apply attention mechanism
    attended_values, attention_weights = attention(Q_proj, K_proj, V_proj)

    return attended_values, attention_weights


def attention(Q, K, V):
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
    )
    dk = Q.size(-1)

    # Compute attention
    scale = torch.sqrt(torch.FloatTensor([dk])).to(device)
    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
    attention_weights = F.softmax(attention_scores, dim=-1)

    # Calculate the weighted sum of values
    attended_values = torch.matmul(attention_weights, V)

    return attended_values, attention_weights


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.W_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        Computes
        FFN(x_i) = ReLU(x_iW_1 + b_1)W_2 + b_2.

        - x has shape batch_size \\times seq_length \\times d_model
        """

        output = self.W_1(x)
        output = self.relu(output)
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
    def __init__(self, d_model, num_heads, d_ffn, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.self_attention = MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.feedforward = PositionwiseFeedForward(d_model, d_ffn, dropout=dropout)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multihead self-attention sub-layer
        attention_output = self.self_attention(x, x, x, mask=mask)
        x = x + self.dropout(attention_output)
        x = self.norm1(x)

        # Position-wise feedforward sub-layer
        ff_output = self.feedforward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x


class DecoderLayer(nn.Module):
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
        self_attention_output = self.self_attention(x, x, x, mask=self_mask)
        x = x + self.dropout(self_attention_output)
        x = self.norm1(x)

        # Encoder-Decoder attention sub-layer
        encoder_attention_output = self.encoder_attention(
            x, encoder_output, encoder_output, mask=encoder_mask
        )
        x = x + self.dropout(encoder_attention_output)
        x = self.norm2(x)

        # Position-wise feedforward sub-layer
        ff_output = self.feedforward(x)
        x = x + self.dropout(ff_output)
        x = self.norm3(x)

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
