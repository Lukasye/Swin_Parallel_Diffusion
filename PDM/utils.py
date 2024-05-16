from PIL import Image
import logging
import os
import datetime

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torch.cuda.amp import GradScaler
import torch.distributed as dist

def collate_fn(batch):
    """Discard none samples.
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def save_images(images: torch.Tensor, save_path: str, unnormalized: bool=False) -> None:
    if unnormalized:
        img = (((images).clamp(-1, 1) + 1 ) * 127.5).type(torch.uint8)
    else:
        img = images
    grid = torchvision.utils.make_grid(img)
    img_arr = grid.permute(1, 2, 0).cpu().numpy()
    img_arr = img_arr.astype(np.uint8)
    img = Image.fromarray(img_arr)
    img.save(save_path)

def save_checkpoint(
        model: nn.Module,
        ema_model: nn.Module,
        optimizer: optim.Optimizer = None,
        scheduler: optim.lr_scheduler = None,
        grad_scaler: GradScaler = None,
) -> None:
    checkpoint = {
        'state_dict': model.state_dict(),
        'ema_state_dict': ema_model.state_dict()
    }
    if optimizer:
        checkpoint['optimizer'] = optimizer.state_dict()
    if scheduler:
        checkpoint['scheduler'] = scheduler.state_dict()
    if scheduler:
        checkpoint['grad_scaler'] = grad_scaler.state_dict()

    return checkpoint


def get_logger(logpath, displaying=True, saving=True):
    logger = logging.getLogger()
    level = logging.INFO

    if (logger.hasHandlers()):
        logger.handlers.clear()

    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        info_file_handler.setFormatter(formatter)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def get_device(rank:int=0):
    if rank == -1:
        return torch.device("cpu")
    return torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")


def resize_tensor(input_tensor, height=299, width=299):
    """ Resize the image to the size for Inception v3 model"""
    output = F.interpolate(input_tensor, size=(height, width), mode='bilinear', align_corners=False)
    output = ((output.clamp(-1, 1) + 1 ) * 127.5).type(torch.uint8)
    return output


def value_range(input: torch.Tensor):
    return torch.min(input).item(), torch.max(input).item()


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # initialize the process group
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30)
    )