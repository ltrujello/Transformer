import torch
import argparse
import torch.nn as nn
import logging
import sys
import datetime
from torch.utils.data import DataLoader, Dataset
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.optim.lr_scheduler import LambdaLR
from transformer.model import Transformer
from transformer.translation import tokens_to_string
from transformer.attention import compute_src_mask, compute_tgt_mask, future_mask
from typing import Optional
import matplotlib.pyplot as plt
from transformer.translation import eval_model


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
        checkpoint_every: Optional[int],
        checkpoint_epochs: bool,
        checkpoint_final: bool,
    ) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.eval_interval: int = eval_interval
        self.checkpoint_every = checkpoint_every
        self.checkpoint_epochs = checkpoint_epochs
        self.checkpoint_final = checkpoint_final

        self.pad_idx: int = src_vocab["<blank>"]
        self.start_idx: int = src_vocab["<sos>"]
        self.end_idx: int = src_vocab["<eos>"]

        # Track training
        self.curr_epoch = 0

    def eval_model_training(self, src, tgt, output):
        with torch.no_grad():
            # Place in evaluation mode
            self.model.eval()

            LOGGER.info("Example Translations:")
            for j in range(min(3, len(src))):  # Print translations for a few examples
                input_sentence = tokens_to_string(
                    src[j],
                    self.src_vocab,
                )
                target_sentence = tokens_to_string(
                    tgt[j],
                    self.tgt_vocab,
                )
                predicted_sentence = tokens_to_string(
                    output.argmax(dim=-1)[j],
                    self.tgt_vocab,
                )
                greedy_translation = greedy_translate(
                    self.model,
                    src[j].unsqueeze(0),  # unsqueeze to add a batch dimension
                    self.start_idx,
                    self.end_idx,
                    self.pad_idx,
                    max_len=30,
                )
                greedy_translation = tokens_to_string(
                    greedy_translation,
                    self.tgt_vocab,
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
            self.optimizer.zero_grad()
            # Create masks
            # encourage model to predict EOS token, so we feed tgt[:, :-1] to forward pass
            tgt_input = tgt[:, :-1]
            src_mask = compute_src_mask(src, self.pad_idx)
            tgt_mask = compute_tgt_mask(tgt_input, self.pad_idx)

            # Forward pass
            output, _ = self.model(src, tgt_input, tgt_mask, src_mask)

            # Calculate loss
            # Output consists of next token preds, so we feed tgt[:, 1:] to loss func
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
                LOGGER.info(
                    f"Epoch {self.curr_epoch}, Iteration {idx}, {idx / len(self.train_dataloader) * 100}% done, Average Loss: {avg_loss}"
                )
                self.eval_model_training(src, tgt, output)

            batch_size = src.size(0)
            if self.checkpoint_every is not None:
                if idx * batch_size % self.checkpoint_every == 0:
                    self.checkpoint_model()

        return total_loss / len(self.train_dataloader)

    def checkpoint_model(self):
        curr_time: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = f"checkpoints/model_{self.curr_epoch}_{curr_time}.pth"
        LOGGER.info(f"Saving model parameters to disk with {filename=}")
        torch.save(self.model.state_dict(), filename)

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
            # Increment epoch counter
            self.curr_epoch += 1
            # Save model parameters to disk
            if self.checkpoint_epochs:
                self.checkpoint_model()

        # Checkpoint model once at the end.
        if self.checkpoint_final:
            self.checkpoint_model()


def train_model(
    num_epochs: int,
    num_batches: int,
    batch_size: int,
    eval_interval: int,
    checkpoint_every: Optional[int],
    checkpoint_file: Optional[str],
    checkpoint_epochs: bool,
    checkpoint_final: bool,
    run_eval_model: bool,
):
    LOGGER.info(
        f"Training model with {num_epochs=} {num_batches} "
        f"{batch_size=} {eval_interval=} {checkpoint_every=} "
        f"{checkpoint_file=} {checkpoint_epochs=} {checkpoint_final=} "
        f"{run_eval_model=}"
    )
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
    if checkpoint_file is not None:
        LOGGER.info(f"Using {checkpoint_file=} for model parameters")
        model.load_state_dict(torch.load(checkpoint_file))

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
        checkpoint_every,
        checkpoint_epochs,
        checkpoint_final,
    )
    trainer.train(num_epochs)

    if run_eval_model:
        eval_model(model, tgt_vocab, test_dataloader)


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
        src_mask = compute_src_mask(src, padding_token)
        enc_output = model.encode(src, src_mask)

        # Initialize the target sequence with the start token
        tgt = torch.full((1, 1), start_token)

        for _ in range(max_len):
            # Generate the next token
            tgt_mask = future_mask(tgt.size(1))
            dec_output, _ = model.decode(tgt, enc_output, tgt_mask, src_mask)
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
        "--log-level",
        choices=["INFO", "DEBUG", "ERROR", "NONE"],
        default="NONE",
        help="Redirect logs to standard output at a certain level",
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        help="Save model parameters after this many training pairs.",
    )
    ap.add_argument(
        "--checkpoint-file",
        help="Continue training model parameters saved in a checkpoint file.",
    )
    ap.add_argument(
        "--checkpoint-epochs",
        default=False,
        type=bool,
        help="Save model parameters at the end of every epoch.",
    )
    ap.add_argument(
        "--checkpoint-final",
        default=False,
        type=bool,
        help="Save model parameters at the very last epoch.",
    )
    ap.add_argument(
        "--eval-model",
        default=True,
        type=bool,
        help="Run model over test set and report BLEU score.",
    )

    args = ap.parse_args()
    if args.log_level != "NONE":
        logging.basicConfig(stream=sys.stdout, level=args.log_level)

    train_model(
        num_epochs=args.num_epochs,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        eval_interval=args.print_every,
        checkpoint_every=args.checkpoint_every,
        checkpoint_file=args.checkpoint_file,
        checkpoint_epochs=args.checkpoint_epochs,
        checkpoint_final=args.checkpoint_final,
        run_eval_model=args.eval_model,
    )


if __name__ == "__main__":
    main()
