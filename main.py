import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from data.dataset import QuAD
from train.fit import fit


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    return device


def main():
    model = torch.load("./model.pth")
    train_ds = QuAD("./data/train_inputs.csv")
    valid_ds = QuAD("./data/dev_inputs.csv")
    
    batch_size = 4
    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
    )
    valid_dl = DataLoader(
        dataset=valid_ds,
        batch_size=batch_size*2,
        shuffle=False,
    )
    
    epochs = 10
    learning_rate = 2e-5
    device = get_device()
    print(f"Training on {device}")
    
    # Start training
    # Define the optimizer
    # Adamを使うと，lossが下がらない場合がある．AdamWを使おう
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Define the loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # freeze the pre-trained layers
    # # learning scheduler
    # # learning schedulerを使うな！lossが下がらなくなる！
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
    fit(
        train_dataloader=train_dl,
        val_dataloader=valid_dl,
        model=model,
        loss_fn=None,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        epochs=epochs,
    )
    
    torch.save(model, "./fine_tuned_model_epoch_10.pth")
    

if __name__ == "__main__":
    main()
    