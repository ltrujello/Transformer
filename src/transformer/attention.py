import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional
import logging
from transformer.utils import configure_device

LOGGER = logging.getLogger(__name__)
device = configure_device()


def future_mask(sequence_length: int):
    """
    Creates a lower-triangular n \\times n matrix
    used to mask future positions
    """
    attn_shape = (1, sequence_length, sequence_length)
    future_mask = (
        torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8).to(device)
    )
    return (future_mask == 0).to(device)


def compute_src_mask(src: torch.tensor, padding_value: int):
    """
    - src is a tensor with shape (batch_size, seq_len)
    - output is a tensor with shape (batch_size, 1, seq_len)
    """
    return (src != padding_value).unsqueeze(1)


def compute_tgt_mask(tgt: torch.tensor, padding_value: Optional[int] = None):
    """
    - tgt is a tensor with shape (batch_size, seq_len)
    - output is a tensor with shape (batch_size, seq_len, seq_len)
    """
    subsequent_mask = future_mask(tgt.size(1))
    if padding_value is None:
        return subsequent_mask
    padding_mask = (tgt != padding_value).unsqueeze(1).to(device)
    return subsequent_mask & padding_mask


def split_heads(Q: torch.tensor, num_heads: int):
    """
    Split the last dimension into (num_heads, head_dim).
    Reshape the tensor to (batch_size, seq_length, num_heads, head_dim)
    and then transpose to get (batch_size, num_heads, seq_length, head_dim).
    """
    batch_size, seq_length, d_model = Q.size()
    head_dim = d_model // num_heads

    # Reshape to separate heads
    Q = Q.view(batch_size, seq_length, num_heads, head_dim)

    # Transpose to get (batch_size, num_heads, seq_length, head_dim)
    Q = Q.transpose(1, 2)

    return Q


class MultiheadAttention(nn.Module):
    """
    Class to compute multihead attention with num_heads-many heads
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
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

    def forward(
        self,
        Q: torch.tensor,
        K: torch.tensor,
        V: torch.tensor,
        mask: Optional[torch.tensor] = None,
    ):
        LOGGER.debug(
            f"Computing multihead attention with {Q.size()=} {K.size()=} {V.size()=}"
            f" with mask.size()={mask.size() if mask is not None else None}"
        )
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        batch_size = Q.size(0)
        d_model = Q.size(-1)

        # Split into multiple heads. Shape should now be (batch_size, num_heads, seq_length, head_dim)
        Q = split_heads(Q, self.num_heads)
        K = split_heads(K, self.num_heads)
        V = split_heads(V, self.num_heads)

        # Add an extra dimension to the mask to get (batch_size, 1, 1, seq_length)
        if mask is not None:
            mask = mask.unsqueeze(1)

        # Compute attention
        output, attention_weights = attention(Q, K, V, dropout=self.dropout, mask=mask)
        # Concatenate the heads and compute transformation
        output = output.permute(0, 2, 1, 3).reshape(batch_size, -1, d_model)
        output = self.W_o(output)

        # if self.training:
        #     return output, None
        return output, attention_weights


def attention(
    Q: torch.tensor,
    K: torch.tensor,
    V: torch.tensor,
    dropout: Optional[nn.Dropout] = None,
    mask: Optional[torch.tensor] = None,
) -> tuple[torch.tensor, Optional[torch.tensor]]:
    """
    Computes attention given query, keys, values.
    If we have n-many key-value pairs of dimension dk, dv respectively
    and m-many queries of dimension dk, then

    - Q has shape (batch_size, m, dk)
    - K has shape (batch_size, n, dk)
    - V has shape (batch_size, n, dv)
    In the transformer architecture we have
    - m = n = seq_len
    - dk = dv = dmodel

    Thus, the attention_weights has shape (batch_size, seq_len, seq_len)
    and the attended_values has shape (batch_size, seq_len, d_model)
    """
    LOGGER.debug(
        f"computing attention with dimensions {Q.size()=} {K.size()=} {V.size()=}"
        f" with mask.size()={mask.size() if mask is not None else None}"
    )
    # Ensure dk is of float type for precision
    dk = float(Q.size(-1))

    # Compute attention
    scale = torch.sqrt(torch.tensor(dk)).to(device)
    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

    # Apply attention mask (if provided).
    if mask is not None:
        LOGGER.debug(f"Applying {mask.size()=} to {attention_scores.size()=}")
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

    attention_weights = F.softmax(attention_scores, dim=-1)
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # Calculate the weighted sum of values
    attended_values = torch.matmul(attention_weights, V)

    return attended_values, attention_weights
