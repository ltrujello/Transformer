import torch
import argparse
import torch.nn as nn
import logging
import sys
import datetime
from pathlib import Path
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
from transformer.translation import eval_model, plot_attention, plot_alignment


LOGGER = logging.getLogger(__name__)
LOGGER_FMT = logging.Formatter(
    "%(levelname)s:%(name)s [%(asctime)s] %(message)s", datefmt="%d/%b/%Y %H:%M:%S"
)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# device = torch.device("cpu")


def build_vocabularies(
    train_dataset: Dataset, test_dataset: Dataset, tokenizer
) -> tuple:
    # Build vocabulary from iterator
    src_vocab = build_vocab_from_iterator(
        map(tokenizer, [x for x, _, _ in (train_dataset + test_dataset)]),
        specials=["<eos>", "<sos>", "<blank>", "<unk>"],
    )
    tgt_vocab = build_vocab_from_iterator(
        map(tokenizer, [x for _, x, _ in (train_dataset + test_dataset)]),
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
    inputs, targets, alignments = zip(*batch)

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

    # create alignment matrix
    batch_size = len(batch)
    src_seq_len = padded_inputs.size(1)
    tgt_seq_len = padded_targets.size(1)
    alignment_data = torch.zeros(batch_size, tgt_seq_len - 1, src_seq_len)
    for idx, alignment in enumerate(alignments):
        alignment_pairs: list[tuple[int]] = [
            tuple(map(int, pair.split("-"))) for pair in alignment.split(" ")
        ]
        for pair in alignment_pairs:
            x, y = pair
            alignment_data[idx][y + 1][x + 1] = 1  # shift by one because of SOS tokens

    return padded_inputs, padded_targets, alignment_data


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
    training_data = []
    for idx, (x, y, z) in enumerate(train_dataset):
        if idx > num_batches * batch_size:
            break
        training_data.append((src_vocab(tokenizer(x)), tgt_vocab(tokenizer(y)), z))

    test_data = []
    for idx, (x, y, z) in enumerate(test_dataset):
        if idx > num_batches * batch_size:
            break
        test_data.append((src_vocab(tokenizer(x)), tgt_vocab(tokenizer(y)), z))

    train_dataloader = DataLoader(
        training_data,
        collate_fn=lambda x: collate_fn(x, pad_idx, start_idx, end_idx),
        batch_size=batch_size,
        shuffle=False,
    )
    test_dataloader = DataLoader(
        test_data,
        collate_fn=lambda x: collate_fn(x, pad_idx, start_idx, end_idx),
        batch_size=batch_size,
        shuffle=False,
    )

    LOGGER.info(
        f"Successfully created training dataloader of size {len(train_dataloader)=} with {train_dataloader.batch_size=} and "
        f"test dataloader of size {len(test_dataloader)=} with {test_dataloader.batch_size=}"
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


def compute_alignment_error(attn_weights, alignment, mse, layer, head):
    """
    attn_weights comes from decoder-encoder attention calculation.
    It is assumed to be a list of shapes (batch_size, num_heads, tgt_seq_len - 1, src_seq_len).
    The length of the list is the number of layers
    Thus, alignment should have shape (batch_size, tgt_seq_len - 1, src_seq_len).

    Note that we subtract 1 from tgt_seq_len because during training, we remove the
    last column of tgt_seq_len before feeding it to the decoder.
    """
    loss = 0
    # Traverse each sentence, translation pair, and compute alignment loss
    for elem in range(len(alignment)):
        # Grab the attention matrix at layer 5, attention head 1
        first_attn_head = attn_weights[layer][elem][head]
        masked_attn_head = first_attn_head * alignment[elem]
        loss += mse(alignment[elem], masked_attn_head)
    return loss


class TranslationDataset(Dataset):
    def __init__(
        self,
        source_file_path,
        target_file_path,
        alignment_file_path,
        split_ratio=1,
        train=True,
        source_transform=None,
        target_transform=None,
    ):
        """
        Args:
            source_file_path (str): Path to the file containing source sentences.
            target_file_path (str): Path to the file containing target sentences.
        """
        self.split_ratio = split_ratio
        self.train = train

        # Read the files and split into lines
        with open(source_file_path) as f:
            self.source_sentences = [line.strip() for line in f.readlines()]

        with open(target_file_path) as f:
            self.target_sentences = [line.strip() for line in f.readlines()]

        with open(alignment_file_path) as f:
            self.alignments = [line.strip() for line in f.readlines()]

        assert (
            len(self.source_sentences)
            == len(self.target_sentences)
            == len(self.alignments)
        ), f"Mismatch in number of sentences between source and target files. {len(self.source_sentences)=} {len(self.target_sentences)=} {len(self.alignments)=}"

        self.source_transform = source_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.source_sentences)

    def __getitem__(self, idx):
        source_sentence = self.source_sentences[idx]
        target_sentence = self.target_sentences[idx]
        alignment = self.alignments[idx]

        if self.source_transform:
            source_sentence = self.source_transform(source_sentence)

        if self.target_transform:
            target_sentence = self.target_transform(target_sentence)

        return source_sentence, target_sentence, alignment


def build_datasets(
    train_source_file_path,
    train_target_file_path,
    train_alignment_file_path,
    test_source_file_path,
    test_target_file_path,
    test_alignment_file_path,
):
    train_dataset = TranslationDataset(
        source_file_path=train_source_file_path,
        target_file_path=train_target_file_path,
        alignment_file_path=train_alignment_file_path,
    )
    test_dataset = TranslationDataset(
        source_file_path=test_source_file_path,
        target_file_path=test_target_file_path,
        alignment_file_path=test_alignment_file_path,
    )
    LOGGER.info(
        f"Successfully created training dataset of size {len(train_dataset)=} and "
        f"test dataset of size {len(test_dataset)=}"
    )
    return train_dataset, test_dataset


class TrainWorker:
    def __init__(
        self,
        model,
        train_dataloader,
        test_dataset,
        tokenizer,
        criterion,
        optimizer,
        lr_scheduler,
        src_vocab,
        tgt_vocab,
        eval_interval,
        root: str,
        checkpoint_every: Optional[int],
        checkpoint_epochs: bool,
        save_model_run: Optional[bool],
        exp_layer: int,
        exp_head: int,
        run_experiment: bool = False,
    ) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.eval_interval: int = eval_interval
        self.checkpoint_every = checkpoint_every
        self.checkpoint_epochs = checkpoint_epochs
        self.save_model_run = save_model_run
        self.epoch_loss: list[float] = []
        self.root = Path(root)
        self.exp_layer = exp_layer
        self.exp_head = exp_head
        self.run_experiment = run_experiment

        if self.save_model_run and not self.root.exists():
            LOGGER.info(f"Creating directory for model details with path {self.root=}")
            self.root.mkdir()

        self.pad_idx: int = src_vocab["<blank>"]
        self.start_idx: int = src_vocab["<sos>"]
        self.end_idx: int = src_vocab["<eos>"]

        self.mse = nn.MSELoss()

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

        for idx, (src, tgt, alignment) in enumerate(self.train_dataloader):
            src = src.to(device)
            tgt = tgt.to(device)
            alignment = alignment.to(device)

            self.optimizer.zero_grad()
            # Create masks
            # encourage model to predict EOS token, so we feed tgt[:, :-1] to forward pass
            tgt_input = tgt[:, :-1]
            src_mask = compute_src_mask(src, self.pad_idx)
            tgt_mask = compute_tgt_mask(tgt_input, self.pad_idx)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask.to(device)

            # Forward pass
            output, attn_weights = self.model(src, tgt_input, tgt_mask, src_mask)

            # Calculate loss
            # Output consists of next token preds, so we feed tgt[:, 1:] to loss func
            tgt_output = tgt[:, 1:]
            loss = self.criterion(
                output.view(-1, output.size(-1)), tgt_output.reshape(-1)
            )
            # Can we guide the attention of the Transformer?
            if self.run_experiment:
                loss += compute_alignment_error(
                    attn_weights, alignment, self.mse, self.exp_layer, self.exp_head
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
        model_name = self.root.name
        filename = str(self.root / f"{model_name}_epoch_{self.curr_epoch}.pth")
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
            # Collect epoch loss
            self.epoch_loss.append(epoch_loss)

            # Increment epoch counter
            self.curr_epoch += 1

            # Save model parameters to disk if applicable
            if self.checkpoint_epochs:
                self.checkpoint_model()

        # Perform various routines on trained model if desired
        if self.save_model_run:
            self.model.eval()
            self.checkpoint_model()
            self.create_training_loss_graph()
            self.sample_and_save_attention()

    def create_training_loss_graph(self):
        """
        Creates a training loss graph. Saves a line plot with epoch num on the x-axis,
        and epoch loss on the y-axis.
        """
        num_epochs = len(self.epoch_loss)
        epochs = range(num_epochs)

        # Create a line plot
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(num_epochs),
            self.epoch_loss,
            marker="o",
            linestyle="-",
            color="b",
            label="Training Loss",
        )
        plt.title(f"Training Loss of {self.root.name} Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.xticks(epochs)  # Set x-ticks to be the epochs
        plt.legend()
        plt.grid(True)

        # Save the plot to a file
        training_loss_filename = str(self.root / "training_loss.png")
        plt.savefig(training_loss_filename)
        LOGGER.info(
            f"Created and saved training loss graph to {training_loss_filename=}"
        )

    def sample_and_save_attention(self):
        """
        Samples 5 sentences from test dataset and plots their attention and word alignments.
        """
        reset_to_training = False
        if self.model.training:
            reset_to_training = True

        self.model.eval()
        # Grab five sentences from the test dataset
        for idx in range(5):
            src, tgt, alignment = self.test_dataset[idx]
            # Preprocess data
            src, tgt, alignment = collate_fn(
                [
                    (
                        self.src_vocab(self.tokenizer(src)),
                        self.tgt_vocab(self.tokenizer(tgt)),
                        alignment,
                    )
                ],
                self.pad_idx,
                self.start_idx,
                self.end_idx,
            )
            src = src.to(device)
            tgt = tgt.to(device)

            # Forward pass
            tgt_input = tgt[:, :-1]
            src_mask = compute_src_mask(src, self.pad_idx)
            tgt_mask = compute_tgt_mask(tgt_input, self.pad_idx)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask.to(device)
            output, attn_weights = self.model(src, tgt_input, tgt_mask, src_mask)

            # Plot attention and save to disk
            plot_attention(
                attention_head_weights=attn_weights,
                sentence_ind=0,
                layer=0,
                attention_head=0,
                src_sentence=tokens_to_string(src[0], self.src_vocab),
                tgt_sentence=tokens_to_string(tgt[0], self.tgt_vocab),
                filename=str(self.root / f"attention_sample_{idx}.png"),
            )

            # Plot word alignment and save to disk
            plot_alignment(
                alignment=alignment[0],
                src_sentence=tokens_to_string(src[0], self.src_vocab),
                tgt_sentence=tokens_to_string(tgt[0], self.tgt_vocab),
                filename=str(self.root / f"alignment_{idx}.png"),
            )

        if reset_to_training:
            self.model.train()


def create_model_name():
    curr_time: str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f"model_{curr_time}"
    return model_name


def train_model(
    num_epochs: int,
    num_batches: int,
    batch_size: int,
    eval_interval: int,
    checkpoint_every: Optional[int],
    checkpoint_file: Optional[str],
    checkpoint_epochs: bool,
    save_model_run: bool,
    run_eval_model: bool,
    message: Optional[str],
    learning_rate: float,
    exp_layer: int,
    exp_head: int,
    root: Optional[str],
    run_experiment: bool = False,
):
    if not root:
        model_name = create_model_name()
    else:
        root = Path(root)
        if not root.exists():
            root.mkdir()
        model_name = root / create_model_name()

    if save_model_run:
        # Create directory for the model
        model_dir = Path(model_name)
        if not model_dir.exists():
            model_dir.mkdir()

        # Attach fileHandler to logger and store logs in the model_dir
        log_filename = f"{model_name}/training.log"
        LOGGER.info(f"Saving logs to {log_filename=}")
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(LOGGER_FMT)
        LOGGER.addHandler(file_handler)

        # Write down the model description message in model_dir
        if message is not None:
            notes_filename = f"{model_name}/notes.txt"
            LOGGER.info(f"Creating a notes file with {notes_filename=}")
            with open(notes_filename, "w") as f:
                f.write(message)

    LOGGER.info(
        f"Training model {model_name=} with {num_epochs=} {num_batches} "
        f"{batch_size=} {eval_interval=} {checkpoint_every=} "
        f"{checkpoint_file=} {checkpoint_epochs=} {save_model_run=} "
        f"{run_eval_model=} {learning_rate=} {run_experiment=} "
        f" {exp_layer=} {exp_head=} {message=} "
    )
    train_dataset, valid_dataset = build_datasets(
        train_source_file_path="data/experiment_1/train.de",
        train_target_file_path="data/experiment_1/train.en",
        train_alignment_file_path="data/experiment_1/train.align",
        test_source_file_path="data/experiment_1/val.de",
        test_target_file_path="data/experiment_1/val.en",
        test_alignment_file_path="data/experiment_1/val.align",
    )
    # Define a simple tokenizer
    tokenizer = get_tokenizer("basic_english")

    # Create vocabularies
    src_vocab, tgt_vocab, pad_idx, start_idx, end_idx = build_vocabularies(
        train_dataset, valid_dataset, tokenizer
    )

    # Create dataloaders
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

    model = Transformer(len(src_vocab), len(tgt_vocab))
    model.to(device)
    if checkpoint_file is not None:
        LOGGER.info(f"Using {checkpoint_file=} for model parameters")
        model.load_state_dict(torch.load(checkpoint_file))

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    criterion.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: lr_schedule(
            step, d_model=model.d_model, warmup_steps=400
        ),
    )
    # Train model
    trainer = TrainWorker(
        model=model,
        train_dataloader=train_dataloader,
        test_dataset=valid_dataset,
        tokenizer=tokenizer,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        eval_interval=eval_interval,
        checkpoint_every=checkpoint_every,
        checkpoint_epochs=checkpoint_epochs,
        save_model_run=save_model_run,
        root=model_name,
        run_experiment=run_experiment,
        exp_layer=exp_layer,
        exp_head=exp_head,
    )

    trainer.train(num_epochs)

    if run_eval_model:
        eval_model(model, tgt_vocab, test_dataloader, LOGGER)

    return model, train_dataloader, test_dataloader, src_vocab, tgt_vocab


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
        src_mask = src_mask.to(device)
        enc_output = model.encode(src, src_mask)

        # Initialize the target sequence with the start token
        tgt = torch.full((1, 1), start_token).to(device)

        for _ in range(max_len):
            # Generate the next token
            tgt_mask = future_mask(tgt.size(1))
            tgt_mask = tgt_mask.to(device)
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
        "--save-model-run",
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
    ap.add_argument(
        "--message",
        type=str,
        help="Include a helpful message describing the model. Only used when save-model-run is True.",
    )
    ap.add_argument(
        "--run-experiment",
        default=False,
        type=bool,
        help="Optionally run experiment.",
    )
    ap.add_argument(
        "--exp-layer",
        type=int,
        help="Layer which we modifiy in our experiment.",
    )
    ap.add_argument(
        "--exp-head",
        type=int,
        help="Attention head which we modify in our experiment.",
    )
    ap.add_argument(
        "--learning-rate",
        default=0.1,
        type=float,
        help="Learning rate for the model.",
    )
    ap.add_argument(
        "--model-root",
        type=str,
        help="Parent directory to store the model.",
    )
    ap.add_argument(
        "--log-stdout",
        default=True,
        type=bool,
        help="Control whether to log to standard output.",
    )

    args = ap.parse_args()
    if args.log_stdout:
        # Create a stream handler for stdout
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(args.log_level)
        stream_handler.setFormatter(LOGGER_FMT)
        LOGGER.addHandler(stream_handler)
        LOGGER.setLevel(args.log_level)

    train_model(
        num_epochs=args.num_epochs,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        eval_interval=args.print_every,
        checkpoint_every=args.checkpoint_every,
        checkpoint_file=args.checkpoint_file,
        checkpoint_epochs=args.checkpoint_epochs,
        save_model_run=args.save_model_run,
        run_eval_model=args.eval_model,
        message=args.message,
        learning_rate=args.learning_rate,
        run_experiment=args.run_experiment,
        exp_layer=args.exp_layer,
        exp_head=args.exp_head,
        root=args.model_root,
    )


if __name__ == "__main__":
    main()
