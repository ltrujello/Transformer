import pytest
import torch
from transformer.transformer import MultiheadAttention, split_heads


@pytest.fixture
def multihead_attention_instance():
    # Assuming you have a MultiheadAttention class
    d_model = 4
    num_heads = 2
    return MultiheadAttention(d_model, num_heads)


def test_split_heads():
    Q = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
            [
                [13.0, 14.0, 15.0, 16.0],
                [17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0],
            ],
        ]
    )
    res = split_heads(Q, 2)
    err = res - torch.tensor(
        [
            [
                [[1.0, 2.0], [5.0, 6.0], [9.0, 10.0]],
                [[3.0, 4.0], [7.0, 8.0], [11.0, 12.0]],
            ],
            [
                [[13.0, 14.0], [17.0, 18.0], [21.0, 22.0]],
                [[15.0, 16.0], [19.0, 20.0], [23.0, 24.0]],
            ],
        ]
    )
    assert torch.sum(err).item() < 1e-16


def test_multihead_attention(multihead_attention_instance):
    # Input tensor dimensions
    batch_size = 2
    sequence_length = 3
    d_model = 4

    # Create an instance of the multihead_attention class
    multihead_attention = multihead_attention_instance

    # Create hardcoded input tensors for testing
    Q = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
            [
                [13.0, 14.0, 15.0, 16.0],
                [17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0],
            ],
        ]
    )

    K = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
            [
                [13.0, 14.0, 15.0, 16.0],
                [17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0],
            ],
        ]
    )

    V = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
            [
                [13.0, 14.0, 15.0, 16.0],
                [17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0],
            ],
        ]
    )

    # Apply multi-head attention
    output, attention_weights = multihead_attention(Q, K, V)

    # Check output dimensions
    assert output.shape == (
        batch_size,
        sequence_length,
        d_model,
    ), "Output shape is incorrect"

    # Check attention weights dimensions
    assert attention_weights.shape == (
        batch_size,
        multihead_attention.num_heads,
        sequence_length,
        sequence_length,
    ), "Attention weights shape is incorrect"


if __name__ == "__main__":
    pytest.main()
