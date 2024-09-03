import numpy as np
import torchmetrics
import torch
from torch import nn

def train_loop(device, dataloader, model, loss_fn, optimizer, batch_index: int, epochs: int, epoch: int, debug: bool=True):
    """
    Trains an epoch of the model

    Parameters:
        - `device`: destination device
        - `dataloader`: the dataloader of the dataset
        - `model`: the model used
        - `loss_fn`: the loss function of the model
        - `optimizer`: the optimizer
        - `batch_index`: the number of the currently processed batch
        - `epochs`: the number of epochs
        - `epoch`: the index of the epoch
        - `debug`: (default `True`): prints debug info
    """
    model.train()
    size = len(dataloader)
    predictions = torch.zeros((batch_size, 75, 38)).to(device)
    labels = torch.zeros((batch_size, 37)).to(device)

    # Get the item from the dataset
    for item, (x, y) in enumerate(dataloader):
        # Move data to the device used
        video = x.to(device)
        label = y.to(device)

        # Compute the prediction and the loss
        pred = model(video)
        predictions[item] = pred
        labels[item] = label

    loss = loss_fn(
        predictions.permute(1, 0, 2),
        labels,
        torch.full(size=(batch_size, ), fill_value=75, dtype=torch.long), # torch.Size([32])
        torch.full(size=(batch_size, ), fill_value=37, dtype=torch.long)  # torch.Size([32])
    )

    # Adjust the weights
    optimizer.step()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.zero_grad()

    if debug: print(f"→ Loss: {loss} [Batch {batch_index + 1}/125, Epoch {epoch + 1}/{epochs}]")

    print(f"===     The batch {batch_index + 1}/160 has finished training     ===")


def GNLAccuracy(preds, labels) -> float:
    alphabet = [x for x in "abcdefghijklmnopqrstuvwxyz0123456789 "]
    total = 0
    for index, video in enumerate(preds):
        correct = 0
        pred_label = []
        label = [i for i in labels[index] if i != " "]
        for frame in video:
            letter = alphabet[torch.argmax(frame)]
            if letter != " ": pred_label.append(letter)

        for i, c in enumerate(pred_label):
            if c == label[i]:
                correct += 1
        total += correct / len(pred_label)
    return total / batch_size


def test_loop(device, dataloader, model, batch_index, loss_fn, debug=True):
    model.eval()
    size = len(dataloader)

    # Disable the updating of the weights
    with torch.no_grad():
        for item, (x, y) in enumerate(dataloader):
            # Move data to the device used
            video = x.to(device)
            label = y.to(device)

            # Compute the prediction and the loss
            pred = model(video)
    print(f"===     The batch {batch_index + 1}/160 has finished training     ===")
