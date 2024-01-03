import torch
import torch.nn.functional as F
import torch.nn as nn
import logging

LOGGER = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_heads(Q, num_heads):
    return torch.stack(Q.split(num_heads, dim=-1), dim=1)


class MultiheadAttention(nn.Module):
    """
    Class to compute multihead attention with n_heads-many heads
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
        LOGGER.info(f"{Q.size()=} {K.size()=} {V.size()=}")

        # Compute attention
        output, attention_weights = attention(Q, K, V)
        # Concatenate the heads and compute transformation
        batches = []
        for batch in output:
            heads = []
            for head in batch:
                heads.append(head)
            batches.append(torch.hstack(heads))
        output = torch.stack(batches)
        output = self.W_o(output)
        LOGGER.info(f"{output=} {output.size()=} {output.shape=}")

        return output, attention_weights


def self_attention(Q, K, V, W_q, W_k, W_v):
    """
    Computes self-attention

    - W_q is shape dmodel \\times d_k
    - W_k is shape dmodel \\times d_k
    - W_v is shape dmodel \\times d_v
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

    - Q is shape m \\times dk
    - K is shape n \\times dk
    - V is shape n \\times dv
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
