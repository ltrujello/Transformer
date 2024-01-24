import torch
import argparse
import torch.nn as nn
import logging
import sys
from torch.utils.data import DataLoader, Dataset
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.optim.lr_scheduler import LambdaLR
from transformer.model import Transformer, future_mask
from typing import Optional
import matplotlib.pyplot as plt


LOGGER = logging.getLogger(__name__)


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

    # Convert tokenized sequences to tensors.
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
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_data,
        collate_fn=lambda x: collate_fn(x, pad_idx, start_idx, end_idx),
        batch_size=batch_size,
        shuffle=True,
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
    padding_mask = (tgt != padding_value).unsqueeze(1)
    return subsequent_mask & padding_mask.view(padding_mask.size(0), -1, 1)


class TrainWorker:
    def __init__(
        self,
        model,
        train_dataloader,
        criterion,
        optimizer,
        lr_scheduler,
        src_vocab,
        tgt_vocab,
        eval_interval,
    ) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.eval_interval: int = eval_interval

        self.pad_idx: int = src_vocab["<blank>"]
        self.start_idx: int = src_vocab["<sos>"]
        self.end_idx: int = src_vocab["<eos>"]

    def eval_model_training(self, src, tgt, output):
        with torch.no_grad():
            # Place in evaluation mode
            self.model.eval()

            LOGGER.info("Example Translations:")
            for j in range(min(3, len(src))):  # Print translations for a few examples
                input_sentence = " ".join(
                    [self.src_vocab.lookup_token(elem.item()) for elem in src[j]]
                )
                target_sentence = " ".join(
                    [self.tgt_vocab.lookup_token(elem.item()) for elem in tgt[j]]
                )
                predicted_sentence = " ".join(
                    [
                        self.tgt_vocab.lookup_token(elem.item())
                        for elem in output.argmax(dim=-1)[j]
                    ]
                )
                greedy_translation = greedy_translate(
                    self.model,
                    src[j].unsqueeze(0),  # unsqueeze to add a batch dimension
                    self.start_idx,
                    self.end_idx,
                    self.pad_idx,
                    max_len=30,
                )
                greedy_translation = " ".join(
                    [self.tgt_vocab.lookup_token(elem) for elem in greedy_translation]
                )
                LOGGER.info(f"Input: {input_sentence}")
                LOGGER.info(f"Target: {target_sentence}")
                LOGGER.info(f"Predicted: {predicted_sentence}")
                LOGGER.info(f"Greedy Translated: {greedy_translation}")

            # Put back to training mode
            self.model.train()

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0

        for idx, (src, tgt) in enumerate(self.train_dataloader):
            # Create masks
            tgt_input = tgt[:, :-1]
            src_mask = compute_src_mask(src, self.pad_idx)
            tgt_mask = compute_tgt_mask(tgt_input, self.pad_idx)

            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(src, tgt_input, tgt_mask, src_mask)

            # Calculate loss
            tgt_output = tgt[:, 1:]
            loss = self.criterion(
                output.view(-1, output.size(-1)), tgt_output.reshape(-1)
            )
            total_loss += loss.item()

            # Backward pass, scheduler and update weights
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            if idx % self.eval_interval == 0:
                avg_loss = total_loss / self.eval_interval
                LOGGER.info(f"Iteration {idx}, Average Loss: {avg_loss}")
                self.eval_model_training(src, tgt, output)
                # self.save_mask_to_disk(src_mask, f"masks/mask_{idx}.png")
        return total_loss / len(self.train_dataloader)

    def save_mask_to_disk(mask):
        # Ensure mask is on CPU and convert to numpy for visualization
        mask = mask.squeeze()
        mask_array = mask.cpu().numpy()

        # Create a heatmap visualization of the mask
        plt.figure(figsize=(10, 10))
        plt.imshow(mask_array, cmap="gray", interpolation="none")
        plt.title("Mask Visualization")
        plt.xlabel("Sequence Position")
        plt.ylabel("Batch Element" if mask_array.shape[0] > 1 else "Sequence Position")
        plt.colorbar(label="Mask Value", orientation="vertical")

        # Save the figure to the specified file path
        # plt.savefig(file_path, bbox_inches="tight")
        plt.close()

    def train(self, num_epochs: int):
        self.model.train()
        for epoch in range(num_epochs):
            LOGGER.info(f"starting {epoch=}")
            epoch_loss = self.train_one_epoch()
            LOGGER.info(f"finished {epoch=} has {epoch_loss=}")


def train_model(num_epochs, num_batches, batch_size, eval_interval):
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
    # Train model
    trainer = TrainWorker(
        model,
        train_dataloader,
        criterion,
        optimizer,
        lr_scheduler,
        src_vocab,
        tgt_vocab,
        eval_interval,
    )
    trainer.train(num_epochs)


def greedy_translate(model, src, start_token, end_token, padding_token, max_len=50):
    """
    Perform greedy translation.
    - src: Input source sequence tensor.
    - src_mask: Mask for the source sequence.
    - max_len: Maximum length of the generated translation.
    - start_token: Index of the start-of-sequence token.
    - end_token: Index of the end-of-sequence token.

    Returns:
    - translated_tokens: List of token indices for the generated translation.
    """
    model.eval()
    with torch.no_grad():
        # Encode the source sequence
        LOGGER.info(f"greedily translating {src.size()=}")
        src_mask = compute_src_mask(src, padding_token)
        enc_output = model.encode(src, src_mask)

        # Initialize the target sequence with the start token
        tgt = torch.full((1, 1), start_token)

        for _ in range(max_len):
            # Generate the next token
            tgt_mask = future_mask(tgt.size(1))
            dec_output = model.decode(tgt, enc_output, tgt_mask, src_mask)
            dec_output = model.output_layer(dec_output)
            next_token = dec_output[:, -1, :].argmax(dim=-1).unsqueeze(1)

            # Append the generated token to the target sequence
            tgt = torch.cat([tgt, next_token], dim=1)

            # Stop if the end token is generated
            if next_token.item() == end_token:
                break

    # Convert tensor to list of token indices
    translated_tokens = tgt.squeeze().tolist()
    return translated_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--num-epochs",
        default=1,
        type=int,
        help="total number of epochs",
    )
    ap.add_argument(
        "--num-batches",
        default=100,
        type=int,
        help="total number of batches to train on",
    )
    ap.add_argument(
        "--batch-size",
        default=10,
        type=int,
        help="size of each batch",
    )
    ap.add_argument(
        "--print-every",
        default=1000,
        type=int,
        help="print loss info every this many training examples",
    )
    ap.add_argument(
        "--log-std",
        default=False,
        action="store_true",
        help="Redirect logs to standard output",
    )

    args = ap.parse_args()
    if args.log_std:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    train_model(
        num_epochs=args.num_epochs,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        eval_interval=args.print_every,
    )


if __name__ == "__main__":
    main()
