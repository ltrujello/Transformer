import pytest
import torch
from transformer.train import compute_tgt_mask
from transformer.model import (
    MultiheadAttention,
    split_heads,
    attention,
    positional_encoding,
    future_mask,
    Transformer,
)


@pytest.fixture
def multihead_attention_instance():
    # Assuming you have a MultiheadAttention class
    d_model = 4
    num_heads = 2
    return MultiheadAttention(d_model, num_heads, dropout=0)


@pytest.fixture
def transformer_instance():
    # Assuming you have a MultiheadAttention class
    num_encoder_stacks = 6
    num_decoder_stacks = 6
    num_encoder_heads = 2
    num_decoder_heads = 2
    src_vocab_size = 100
    tgt_vocab_size = 100
    d_model = 4
    d_ff = 10
    max_seq_len = 10

    return Transformer(
        num_encoder_stacks=num_encoder_stacks,
        num_decoder_stacks=num_decoder_stacks,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        d_ffn=d_ff,
        num_encoder_heads=num_encoder_heads,
        num_decoder_heads=num_decoder_heads,
        max_seq_len=max_seq_len,
    )


@pytest.fixture
def query_key_value():
    Q = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
            [
                [13.0, 14.0, 15.0, 16.0],
                [17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0],
                [25.0, 26.0, 27.0, 28.0],
            ],
        ]
    )

    K = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
            [
                [13.0, 14.0, 15.0, 16.0],
                [17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0],
                [25.0, 26.0, 27.0, 28.0],
            ],
        ]
    )

    V = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
            [
                [13.0, 14.0, 15.0, 16.0],
                [17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0],
                [25.0, 26.0, 27.0, 28.0],
            ],
        ]
    )

    return Q, K, V


@pytest.fixture
def mock_positional_encoding():
    return torch.tensor(
        [
            [0.0000, 1.0000, 0.0000, 1.0000],
            [0.8415, 0.5403, 0.0100, 0.9999],
            [0.9093, -0.4161, 0.0200, 0.9998],
            [0.1411, -0.9900, 0.0300, 0.9996],
        ]
    )


def test_split_heads(query_key_value):
    Q, _, _ = query_key_value
    res = split_heads(Q, 2)
    err = res - torch.tensor(
        [
            [
                [[1.0, 2.0], [5.0, 6.0], [9.0, 10.0], [13.0, 14.0]],  # Head 1, batch 1
                [[3.0, 4.0], [7.0, 8.0], [11.0, 12.0], [15.0, 16.0]],  # Head 2, batch 1
            ],
            [
                [
                    [13.0, 14.0],  # Head 1, batch 2
                    [17.0, 18.0],
                    [21.0, 22.0],
                    [25.0, 26.0],
                ],
                [
                    [15.0, 16.0],  # Head 2, batch 2
                    [19.0, 20.0],
                    [23.0, 24.0],
                    [27.0, 28.0],
                ],
            ],
        ]
    )
    assert torch.sum(err).item() < 1e-16


def test_multihead_attention_dims(multihead_attention_instance, query_key_value):
    # Input tensor dimensions
    batch_size = 2
    sequence_length = 4
    d_model = 4
    multihead_attention_instance.eval()

    # Create an instance of the multihead_attention class
    multihead_attention = multihead_attention_instance

    # Apply multi-head attention
    Q, K, V = query_key_value
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


def test_multihead_attention_calc(multihead_attention_instance, query_key_value):
    multihead_attention = multihead_attention_instance

    # Apply multi-head attention
    Q, K, V = query_key_value
    output, attention_weights = multihead_attention.forward(Q, K, V)

    batch_1_attn_1, _ = attention(
        split_heads(multihead_attention.W_q(Q), 2)[0][0],
        split_heads(multihead_attention.W_k(K), 2)[0][0],
        split_heads(multihead_attention.W_v(V), 2)[0][0],
    )
    batch_1_attn_2, _ = attention(
        split_heads(multihead_attention.W_q(Q), 2)[0][1],
        split_heads(multihead_attention.W_k(K), 2)[0][1],
        split_heads(multihead_attention.W_v(V), 2)[0][1],
    )

    batch_2_attn_1, _ = attention(
        split_heads(multihead_attention.W_q(Q), 2)[1][0],
        split_heads(multihead_attention.W_k(K), 2)[1][0],
        split_heads(multihead_attention.W_v(V), 2)[1][0],
    )
    batch_2_attn_2, _ = attention(
        split_heads(multihead_attention.W_q(Q), 2)[1][1],
        split_heads(multihead_attention.W_k(K), 2)[1][1],
        split_heads(multihead_attention.W_v(V), 2)[1][1],
    )
    print(output[0], batch_1_attn_1)

    assert (
        torch.sum(
            output[0]
            - multihead_attention.W_o(torch.hstack([batch_1_attn_1, batch_1_attn_2]))
        ).item()
        < 1e-5
    )
    assert (
        torch.sum(
            output[1]
            - multihead_attention.W_o(torch.hstack([batch_2_attn_1, batch_2_attn_2]))
        ).item()
        < 1e-5
    )


def test_positional_encoding(mock_positional_encoding):
    encoding = positional_encoding(4, 4)
    print(encoding)
    assert torch.sum(encoding - mock_positional_encoding) < 1e-6


def test_future_mask():
    assert torch.all(
        torch.tensor(
            [
                [
                    [True, False, False, False],
                    [True, True, False, False],
                    [True, True, True, False],
                    [True, True, True, True],
                ]
            ]
        )
        == future_mask(4)
    ), "Upper triangular masking failed"


def test_attention_future_masking():
    batch_size = 2
    seq_len = 10
    d_model = 5
    Q = torch.rand(size=(batch_size, seq_len, d_model))
    K = torch.rand(size=(batch_size, seq_len, d_model))
    V = torch.rand(size=(batch_size, seq_len, d_model))
    mask = future_mask(seq_len)
    attn_vals, attn_weights = attention(Q, K, V, mask=mask)
    upper_triangular = attn_weights.squeeze(1).triu(diagonal=1)
    assert torch.all(upper_triangular == 0), "Attention weights aren't masked"


def test_attention_padding_masking():
    batch_size = 2
    seq_len = 10
    d_model = 5
    Q = torch.rand(size=(batch_size, seq_len, d_model))
    K = torch.rand(size=(batch_size, seq_len, d_model))
    V = torch.rand(size=(batch_size, seq_len, d_model))
    mask = torch.ones(batch_size, 1, seq_len).bool()
    mask[:, :, -1] = False
    mask[:, :, 5] = False

    attn_vals, attn_weights = attention(Q, K, V, mask=mask)

    assert torch.all(attn_weights[:, :, -1] == 0)
    assert torch.all(attn_weights[:, :, 5] == 0)


def test_attention_future_and_padding_masking():
    batch_size = 2
    seq_len = 10
    d_model = 5
    Q = torch.rand(size=(batch_size, seq_len, d_model))
    mask = compute_tgt_mask(torch.rand(size=(batch_size, seq_len)), padding_value=0)
    mask[:, :, -1] = False
    mask[:, :, 5] = False

    attn_vals, attn_weights = attention(Q, Q, Q, mask=mask)

    assert torch.all(attn_weights[:, :, -1] == 0)
    assert torch.all(attn_weights[:, :, 5] == 0)


def test_multihead_attention_future_masking(multihead_attention_instance):
    batch_size = 5
    seq_len = 10
    d_model = 4
    multihead_attention_instance.eval()

    Q = torch.rand(size=(batch_size, seq_len, d_model))
    K = torch.rand(size=(batch_size, seq_len, d_model))
    V = torch.rand(size=(batch_size, seq_len, d_model))
    mask = future_mask(seq_len)

    attn_vals, attn_weights = multihead_attention_instance(Q, K, V, mask=mask)
    upper_triangular = attn_weights.squeeze(1).triu(diagonal=1)
    assert torch.all(upper_triangular == 0), "Attention weights aren't masked"


def test_multihead_attention_padding_masking(multihead_attention_instance):
    batch_size = 5
    seq_len = 10
    d_model = 4
    multihead_attention_instance.eval()

    Q = torch.rand(size=(batch_size, seq_len, d_model))
    K = torch.rand(size=(batch_size, seq_len, d_model))
    V = torch.rand(size=(batch_size, seq_len, d_model))
    mask = torch.ones(batch_size, 1, seq_len)
    mask[:, :, -1] = 0
    mask[:, :, 5] = 0

    attn_vals, attn_weights = multihead_attention_instance(Q, K, V, mask=mask)
    print(attn_weights)

    assert torch.all(attn_weights[:, :, :, -1] == 0)
    assert torch.all(attn_weights[:, :, :, 5] == 0)


def test_transformer_smoke_test(transformer_instance):
    batch_size = 2
    sequence_length = 10
    input = torch.randint(high=10, size=(batch_size, sequence_length))
    tgt = torch.randint(high=10, size=(batch_size, sequence_length))

    src_mask = torch.randint(0, 2, size=(batch_size, 1, sequence_length)).bool()
    tgt_mask = torch.randint(0, 2, size=(1, sequence_length, sequence_length)).bool()
    transformer_instance.forward(input, tgt, src_mask, tgt_mask)
    assert True


def test_transformer_fake_data():
    test_model = Transformer(
        num_encoder_stacks=6, num_decoder_stacks=6, src_vocab_size=11, tgt_vocab_size=11
    )
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10).bool()

    tgt = torch.zeros(1, 1).type_as(src)
    tgt_mask = future_mask(tgt.size(-1))

    for i in range(9):
        print(i)
        prob, _ = test_model.forward(
            src=src,
            tgt=tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
        )
        _, next_word = torch.max(prob[:, -1], dim=1)
        next_word = next_word.data[0]
        print(next_word)
        tgt = torch.cat(
            [tgt, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", tgt)


if __name__ == "__main__":
    pytest.main()
