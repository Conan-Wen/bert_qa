import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def evaluate(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: str,
    writer: SummaryWriter,
    epoch:int
) -> None:
    data_size = len(dataloader.dataset)
    
    model.eval()
    model.to(device=device)
    
    test_loss = 0.0
    
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            # 訓練データをbatchから取り出す
            input_ids, attention_mask = X
            start_positions, end_positions = y[0].squeeze(dim=-1),  y[1].squeeze(dim=-1)
            
            input_ids, attention_mask, start_positions, end_positions = \
            input_ids.to(device=device), attention_mask.to(device=device), start_positions.to(device=device), end_positions.to(device=device)
            
            # 順伝搬
            outputs = model(input_ids, attention_mask, start_positions=start_positions, end_positions=end_positions)
            # start_loss = loss_fn(outputs.start_logits, start_positions)
            # end_loss = loss_fn(outputs.end_logits, end_positions)
            
            test_loss += outputs.loss.item()

    
    test_loss /= data_size
    
    print(f"Validation Error: loss: {test_loss:>7f}\n")
    
    writer.add_scalar(
        tag="val loss",
        scalar_value=test_loss,
        global_step=epoch,
    )
