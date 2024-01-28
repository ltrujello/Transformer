import torch
from transformer.attention import compute_src_mask, compute_tgt_mask
from transformer.train import collate_fn


def test_collate_fn():
    padding_value = 0
    start_value = 1
    end_value = 2
    batch = [
        ([3, 4, 5, 6], [7, 8, 9, 10]),
        ([3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16, 17]),
    ]

    inputs, outputs = collate_fn(batch, padding_value, start_value, end_value)

    assert torch.all(
        inputs
        == torch.tensor(
            [
                [1, 3, 4, 5, 6, 2, 0, 0],
                [1, 3, 4, 5, 6, 7, 8, 2],
            ]
        )
    )
    assert torch.all(
        outputs
        == torch.tensor(
            [
                [1, 7, 8, 9, 10, 2, 0, 0, 0, 0, 0],
                [1, 9, 10, 11, 12, 13, 14, 15, 16, 17, 2],
            ]
        )
    )


def test_compute_src_mask_no_padding():
    src = torch.tensor([[1, 2, 3], [4, 5, 6]])
    padding_value = 0
    src_mask = torch.tensor([[[True, True, True]], [[True, True, True]]])
    assert torch.equal(compute_src_mask(src, padding_value), src_mask)


def test_compute_src_mask_with_padding():
    src = torch.tensor([[1, 0, 3], [4, 0, 6]])
    padding_value = 0
    src_mask = torch.tensor([[[True, False, True]], [[True, False, True]]])

    assert torch.equal(compute_src_mask(src, padding_value), src_mask)


def test_compute_src_mask_batches_with_padding():
    src = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 2], [10, 11, 12]]])
    padding_value = 2
    src_mask = torch.tensor(
        [
            [[[True, False, True], [True, True, True]]],
            [[[True, True, False], [True, True, True]]],
        ]
    )
    assert torch.equal(compute_src_mask(src, padding_value), src_mask)


def test_compute_tgt_mask_no_padding():
    tgt = torch.tensor([[1, 2, 3], [4, 5, 6]])
    padding_value = 0
    tgt_mask = torch.tensor(
        [
            [[True, False, False], [True, True, False], [True, True, True]],
            [[True, False, False], [True, True, False], [True, True, True]],
        ]
    )
    assert torch.equal(compute_tgt_mask(tgt, padding_value), tgt_mask)


def test_compute_tgt_mask_with_padding():
    tgt = torch.tensor(
        [
            [1, 2, 3, 0, 0],  # Sequence 1 (length 3, with padding)
            [1, 4, 5, 6, 0],  # Sequence 2 (length 4, with padding)
        ]
    )
    padding_value = 0
    tgt_mask = torch.tensor(
        [
            [
                [True, False, False, False, False],
                [True, True, False, False, False],
                [True, True, True, False, False],
                [True, True, True, False, False],
                [True, True, True, False, False],
            ],
            [
                [True, False, False, False, False],
                [True, True, False, False, False],
                [True, True, True, False, False],
                [True, True, True, True, False],
                [True, True, True, True, False],
            ],
        ]
    )
    assert torch.equal(compute_tgt_mask(tgt, padding_value), tgt_mask)
