import os
import torch

from dotenv import load_dotenv


def configure_device():
    load_dotenv()
    device = os.getenv("DEVICE", "cpu")
    return torch.device(device)
