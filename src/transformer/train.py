from transformer.transformer import future_mask

import torchtext
import torch
from torch.utils.data import DataLoader, Dataset
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Load the Multi30k dataset
train_dataset, valid_dataset = Multi30k(
    root="data", split=("train", "valid"), language_pair=("de", "en")
)

# Define a simple tokenizer
tokenizer = get_tokenizer("basic_english")


def build_vocabularies(train_dataset, test_dataset):
    # Build vocabulary from iterator
    src_vocab = build_vocab_from_iterator(
        map(tokenizer, [x for x, _ in (train_dataset + test_dataset)]),
        specials=["<eos>", "<sos>", "<blank>", "<unk>"],
    )
    tgt_vocab = build_vocab_from_iterator(
        map(tokenizer, [x for _, x in (train_dataset + test_dataset)]),
        specials=["<eos>", "<sos>", "<blank>", "<unk>"],
    )
    pad_idx = src_vocab["<blank>"]
    start_idx = src_vocab["<sos>"]
    end_idx = src_vocab["<eos>"]

    return src_vocab, tgt_vocab, pad_idx, start_idx, end_idx


def collate_fn(batch: list[tuple], padding_value, start_value, end_value):
    """
    Given a list of tokenized sequences, we add start, end, padding tokens, and
    return a torch.tensor representing the batch.
    """
    print(batch)
    inputs, targets = zip(*batch)

    # Convert tokenized sequences to tensors. 35-40L cotopaxi
    tensor_inputs = [torch.tensor([start_value] + x + [end_value]) for x in inputs]
    tensor_targets = [torch.tensor([start_value] + x + [end_value]) for x in targets]

    # Pad sequences to the maximum length in the batch
    padded_inputs = torch.nn.utils.rnn.pad_sequence(
        tensor_inputs, batch_first=True, padding_value=padding_value
    )
    padded_targets = torch.nn.utils.rnn.pad_sequence(
        tensor_targets, batch_first=True, padding_value=padding_value
    )

    return padded_inputs, padded_targets


src_vocab, tgt_vocab, pad_idx, start_idx, end_idx = build_vocabularies(train_dataset, valid_dataset)
train_dataloader = DataLoader(
    [(src_vocab(tokenizer(x)), tgt_vocab(tokenizer(y))) for x, y in train_dataset],
    collate_fn=lambda x: collate_fn(x, pad_idx, start_idx, end_idx),
)
test_dataloader = DataLoader(
    [(src_vocab(tokenizer(x)), tgt_vocab(tokenizer(y))) for x, y in valid_dataset],
    collate_fn=lambda x: collate_fn(x, pad_idx, start_idx, end_idx),
)


def lr_schedule(step_num, d_model, warmup_steps):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step_num == 0:
        step_num = 1
    return d_model ** (-0.5) * min(
        step_num ** (-0.5), step_num * warmup_steps ** (-1.5)
    )


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, padding_value):
    model.train()
    total_loss = 0.0

    for src, tgt in dataloader:
        # Create masks
        src_mask = (src != padding_value).unsqueeze(1)
        tgt_mask = future_mask(tgt.size(1))

        optimizer.zero_grad()

        # Forward pass
        output = model(src, tgt, tgt_mask, src_mask)

        # Calculate loss
        loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
        total_loss += loss.item()

        # Backward pass, scheduler and update weights 
        loss.backward()
        scheduler.step()
        optimizer.step()

    return total_loss / len(dataloader)
