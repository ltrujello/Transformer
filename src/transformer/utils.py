import os
import torch
import logging
import sys

from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)


def configure_device():
    load_dotenv()
    device = os.getenv("DEVICE", "cpu")
    print(f"Attempting to use device {device=}")
    return torch.device(device)


def configure_logging(log_file_path):
    LOGGER_FMT = logging.Formatter(
        "%(levelname)s:%(name)s [%(asctime)s] %(message)s", datefmt="%d/%b/%Y %H:%M:%S"
    )
    # Create a filehandler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(LOGGER_FMT)

    # Create a stream handler for stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(LOGGER_FMT)

    # Configure parent logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
