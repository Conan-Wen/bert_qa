import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn


def train(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler | None,
    device: str,
    writer: SummaryWriter,
    epoch: int,
) -> None:
    data_size = len(dataloader.dataset)
    
    model.train()
    model.to(device=device)
    
    running_loss = 0.0
    
    for batch, (X, y) in enumerate(dataloader):
        # 訓練データをbatchから取り出す
        input_ids, attention_mask = X
        start_positions, end_positions = y[0].squeeze(dim=-1),  y[1].squeeze(dim=-1)
        
        input_ids, attention_mask, start_positions, end_positions = \
        input_ids.to(device=device), attention_mask.to(device=device), start_positions.to(device=device), end_positions.to(device=device)
        
        # 順伝搬
        outputs = model(input_ids, attention_mask, start_positions=start_positions, end_positions=end_positions)
        
        if loss_fn:
            start_loss = loss_fn(outputs.start_logits, start_positions)
            end_loss = loss_fn(outputs.end_logits, end_positions)
            
            loss = (start_loss + end_loss) / 2.0
        else:
            loss = outputs.loss
        
        # 勾配をリセット
        optimizer.zero_grad()
        
        # # 逆伝搬
        loss.backward()
        
        # パラメータ更新
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        running_loss += loss.item()
        
        # 訓練中のログ
        num_ite = 2
        if (batch + 1) % num_ite == 0:
            # 進度
            current = (batch + 1) * len(X)
            
            running_loss /= num_ite
            
            print(f"loss: {running_loss:>7f}    [{current:>8d}/{data_size:>8d}]")
            
            writer.add_scalar(
                tag="training loss",
                scalar_value=running_loss,
                global_step= epoch * len(dataloader) + (batch+1),
            )
            
            running_loss = 0.0
        