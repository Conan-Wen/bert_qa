import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from train.train import train
from train.evaluate import evaluate

def fit(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    device: str,
    epochs: int = 25,
):
    writer = SummaryWriter(
        log_dir="./runs"
    )
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n -------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, scheduler, device, writer, epoch)
        evaluate(val_dataloader, model, loss_fn, device, writer, epoch)
        
    print("Done!")