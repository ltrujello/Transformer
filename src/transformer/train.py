import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.optim.lr_scheduler import LambdaLR
from transformer.transformer import Transformer, future_mask


def build_vocabularies(
    train_dataset: Dataset, test_dataset: Dataset, tokenizer
) -> tuple:
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


def collate_fn(
    batch: list[tuple], padding_value: int, start_value: int, end_value: int
) -> tuple[torch.tensor, torch.tensor]:
    """
    Given a list of tokenized sequences, we add start, end, padding tokens, and
    return a torch.tensor representing the batch.
    """
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


def build_dataloaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    tokenizer,
    src_vocab,
    tgt_vocab,
    pad_idx,
    start_idx,
    end_idx,
    batch_size,
    num_batches,
) -> tuple[DataLoader, DataLoader]:
    # if num_elements is None:
    #     train_dataloader = DataLoader(
    #         [
    #             (src_vocab(tokenizer(x)), tgt_vocab(tokenizer(y)))
    #             for x, y in train_dataset
    #         ],
    #         collate_fn=lambda x: collate_fn(x, pad_idx, start_idx, end_idx),
    #         batch_size=batch_size,
    #     )
    #     test_dataloader = DataLoader(
    #         [
    #             (src_vocab(tokenizer(x)), tgt_vocab(tokenizer(y)))
    #             for x, y in test_dataset
    #         ],
    #         collate_fn=lambda x: collate_fn(x, pad_idx, start_idx, end_idx),
    #         batch_size=batch_size,
    #     )
    # else:
    training_data = []
    for idx, (x, y) in enumerate(train_dataset):
        if idx > num_batches * batch_size:
            break
        training_data.append((src_vocab(tokenizer(x)), tgt_vocab(tokenizer(y))))

    test_data = []
    for idx, (x, y) in enumerate(test_dataset):
        if idx > num_batches * batch_size:
            break
        test_data.append((src_vocab(tokenizer(x)), tgt_vocab(tokenizer(y))))

    train_dataloader = DataLoader(
        training_data,
        collate_fn=lambda x: collate_fn(x, pad_idx, start_idx, end_idx),
        batch_size=batch_size,
    )
    test_dataloader = DataLoader(
        test_data,
        collate_fn=lambda x: collate_fn(x, pad_idx, start_idx, end_idx),
        batch_size=batch_size,
    )

    return train_dataloader, test_dataloader


def lr_schedule(step_num: int, d_model: int, warmup_steps: int) -> float:
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step_num == 0:
        step_num = 1
    return d_model ** (-0.5) * min(
        step_num ** (-0.5), step_num * warmup_steps ** (-1.5)
    )


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    lr_scheduler,
    padding_value,
    src_vocab,
    tgt_vocab,
    eval_interval=100,
):
    model.train()
    total_loss = 0.0

    for idx, (src, tgt) in enumerate(dataloader):
        # Create masks
        src_mask = (src != padding_value).unsqueeze(1)
        tgt_mask = future_mask(tgt.size(1)) & (tgt != padding_value).unsqueeze(1)

        optimizer.zero_grad()

        # Forward pass
        output = model(src, tgt, tgt_mask, src_mask)

        # Calculate loss
        loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
        total_loss += loss.item()

        # Backward pass, scheduler and update weights
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if idx % eval_interval == 0:
            # Place in evaluation mode
            model.eval()
            avg_loss = total_loss / eval_interval
            print(f"Iteration {idx}, Average Loss: {avg_loss}")

            print("Example Translations:")
            for j in range(min(3, len(src))):  # Print translations for a few examples
                input_sentence = " ".join(
                    [src_vocab.lookup_token(elem.item()) for elem in src[j]]
                )
                target_sentence = " ".join(
                    [tgt_vocab.lookup_token(elem.item()) for elem in tgt[j]]
                )
                predicted_sentence = " ".join(
                    [
                        tgt_vocab.lookup_token(elem.item())
                        for elem in output.argmax(dim=-1)[j]
                    ]
                )
                print(f"Input: {input_sentence}")
                print(f"Target: {target_sentence}")
                print(f"Predicted: {predicted_sentence}")

            # Put back to training mode
            model.train()
    return total_loss / len(dataloader)


def train_model(num_epochs, num_batches, batch_size, eval_interval):
    # Load the Multi30k dataset
    train_dataset, valid_dataset = Multi30k(
        root="data", split=("train", "valid"), language_pair=("de", "en")
    )
    # Define a simple tokenizer
    tokenizer = get_tokenizer("basic_english")

    # Create vocabularies
    src_vocab, tgt_vocab, pad_idx, start_idx, end_idx = build_vocabularies(
        train_dataset, valid_dataset, tokenizer
    )

    # Instantiate model and dataloaders
    model = Transformer(len(src_vocab), len(tgt_vocab))
    train_dataloader, test_dataloader = build_dataloaders(
        train_dataset,
        valid_dataset,
        tokenizer,
        src_vocab,
        tgt_vocab,
        pad_idx,
        start_idx,
        end_idx,
        batch_size=batch_size,
        num_batches=num_batches,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.1, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: lr_schedule(
            step, d_model=model.d_model, warmup_steps=400
        ),
    )

    model.train()
    for epoch in range(num_epochs):
        print(f"{epoch=}")
        epoch_loss = train_one_epoch(
            model,
            train_dataloader,
            criterion,
            optimizer,
            lr_scheduler,
            pad_idx,
            src_vocab,
            tgt_vocab,
            eval_interval,
        )
        print(f"{epoch=} has {epoch_loss=}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--n_epochs",
        default=1,
        type=int,
        help="total number of epochs",
    )
    ap.add_argument(
        "--num_batches",
        default=100,
        type=int,
        help="total number of batches to train on",
    )
    ap.add_argument(
        "--batch_size",
        default=10,
        type=int,
        help="size of each batch",
    )
    ap.add_argument(
        "--print_every",
        default=1000,
        type=int,
        help="print loss info every this many training examples",
    )
    args = ap.parse_args()
    train_model(
        num_epochs=args.num_epochs,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        eval_interval=args.print_every,
    )


if __name__ == "__main__":
    main()
